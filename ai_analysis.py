import os
import time
import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from config import BotConfig
from symbol_state import SymbolState

load_dotenv()
console = Console()


@dataclass
class AISignal:
    """Structured signal output with validation."""
    bias: str = "NEUTRAL"
    confidence: int = 0
    action: str = "WAIT"
    entry: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    reason: str = ""
    model: str = ""
    # FIX #1: distinguish "the model answered WAIT" from "the API call failed".
    # Before, an API error produced a NEUTRAL/WAIT signal that looked like a
    # legitimate opinion, so the disagreement branch could treat one dead model
    # + one live model as a "consensus" and trade on a single model without
    # anyone noticing.
    ok: bool = True

    def to_dict(self) -> dict:
        return {
            "bias": self.bias,
            "confidence": self.confidence,
            "action": self.action,
            "entry": self.entry,
            "sl": self.sl,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "reason": self.reason,
            "model": self.model,
            "ok": self.ok,
        }


class AIAnalyst:
    """
    Dual AI Consensus Engine with Market Regime Awareness.
    - ChatGPT (OpenAI API)
    - Grok (xAI API)
    - Adaptive risk filtering with regime detection
    - Weighted consensus with confidence scoring
    """

    def __init__(self, cfg: BotConfig, executor: ThreadPoolExecutor):
        self.cfg = cfg
        self.executor = executor

        # SEPARATE clients for each AI
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1",
        )

        self.xai_client = OpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1",
        )

        # Signal quality tracking for adaptive thresholds
        self.signal_history: list[dict] = []
        self.adaptive_conf_threshold = 75  # starts lower, adapts up

    # ---------------- SMALL HELPERS ----------------
    @staticmethod
    def _safe_float(v) -> float:
        """Coerce anything (None, '', 'abc', '1.5') to a float without raising."""
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _safe_int(v) -> int:
        """Coerce to int, tolerating floats and numeric strings like '82.5'."""
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return 0

    # ---------------- MARKET REGIME DETECTION ----------------
    def _detect_regime(self, s: SymbolState) -> str:
        """Classify market regime for adaptive filtering."""
        if s.adx >= 30 and s.bb_squeeze:
            return "TRENDING_SQUEEZE"  # explosive move likely
        elif s.adx >= 25:
            return "TRENDING"
        elif s.adx < 20 and s.bb_squeeze:
            return "RANGING_SQUEEZE"  # low vol, avoid
        else:
            return "RANGING"

    def _regime_adjusted_thresholds(self, regime: str) -> dict:
        """Get adaptive thresholds based on market regime."""
        thresholds = {
            "TRENDING_SQUEEZE": {
                "min_confidence": 70,
                "min_adx": 20,
                "min_rr": 1.5,
                "description": "Trending + squeeze = high conviction needed but lower bar"
            },
            "TRENDING": {
                "min_confidence": 75,
                "min_adx": 22,
                "min_rr": 2.0,
                "description": "Standard trending market"
            },
            "RANGING": {
                "min_confidence": 80,
                "min_adx": 15,
                "min_rr": 2.5,
                "description": "Range-bound: need higher confidence, better R:R"
            },
            "RANGING_SQUEEZE": {
                "min_confidence": 85,
                "min_adx": 18,
                "min_rr": 3.0,
                "description": "Low vol squeeze: very selective"
            }
        }
        return thresholds.get(regime, thresholds["TRENDING"])

    # ---------------- SAFE PARSE WITH SCHEMA ----------------
    def _extract_json(self, text: str) -> dict:
        """Robust JSON extraction with regex fallback."""
        if not text:
            return {}

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Extract JSON from markdown code blocks
        patterns = [
            r'```(?:json)?\s*(\{.*?\})\s*```',
            r'```(?:json)?\s*(\[.*?\])\s*```',
            r'\{[^{}]*"bias"[^{}]*\}',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        # Last resort: find first { and last }
        try:
            start = text.index('{')
            end = text.rindex('}') + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return {}

    def _safe_parse(self, text: str, model_name: str) -> AISignal:
        """Parse AI response into structured signal with defaults."""
        data = self._extract_json(text)

        # Validate and sanitize. NOTE: use `or` guards because a JSON null
        # comes back as None, and None.upper() / None[:200] would crash.
        bias = str(data.get("bias") or "NEUTRAL").upper()
        if bias not in {"BULLISH", "BEARISH", "NEUTRAL"}:
            bias = "NEUTRAL"

        action = str(data.get("action") or "WAIT").upper()
        if action not in {"LONG", "SHORT", "WAIT"}:
            action = "WAIT"

        confidence = max(0, min(100, self._safe_int(data.get("confidence", 0))))

        return AISignal(
            bias=bias,
            confidence=confidence,
            action=action,
            entry=self._safe_float(data.get("entry", 0)),
            sl=self._safe_float(data.get("sl", 0)),
            tp1=self._safe_float(data.get("tp1", 0)),
            tp2=self._safe_float(data.get("tp2", 0)),
            reason=str(data.get("reason") or "no reason provided")[:200],
            model=model_name,
            ok=bool(data),  # FIX #1: empty/unparseable response counts as failed
        )

    # ---------------- ENHANCED PROMPT WITH FEW-SHOT ----------------
    def _build_prompt(self, s: SymbolState) -> str:
        regime = self._detect_regime(s)

        return f"""<system>
You are an elite quantitative crypto futures trader with 15+ years experience.
You specialize in high-probability setups with strict risk management.
Market Regime: {regime}
</system>

<task>
Analyze the provided market data and return a trading signal as valid JSON.
</task>

<market_data>
Symbol: {s.symbol}
Current Price: {s.price}
Timeframe: 15m primary, 1h confirmation

Technical Indicators:
- RSI (15m/1h): {s.rsi_15:.2f} / {s.rsi_1h:.2f}
- SMA20: {s.sma20:.4f}
- EMA50: {s.ema50:.4f}
- EMA200: {s.ema200:.4f}
- MACD: {s.macd:.6f}
- MACD Signal: {s.macd_sig:.6f}
- ATR (volatility): {s.atr:.4f}
- ADX (trend strength): {s.adx:.2f}
- Volume OK: {s.volume_ok}
- Bollinger Band Squeeze: {s.bb_squeeze}
</market_data>

<analysis_framework>
1. TREND: Price vs EMA50/EMA200. Bullish if price > EMA50 > EMA200.
2. MOMENTUM: RSI not overbought (>70) or oversold (<30). MACD crossover?
3. VOLATILITY: ATR expansion? BB squeeze suggests imminent move.
4. VOLUME: Confirms trend strength?
5. RISK: Calculate SL at recent swing low/high or ATR-based. Entry at current or pullback.
</analysis_framework>

<examples>
<example>
Input: RSI 65/55, Price > EMA50 > EMA200, MACD bullish crossover, ADX 32, Volume OK
Output: {{"bias":"BULLISH","confidence":82,"action":"LONG","entry":45000,"sl":44200,"tp1":46500,"tp2":48000,"reason":"Bullish trend continuation with momentum confirmation"}}
</example>
<example>
Input: RSI 45/40, Price between EMA50 and EMA200, ADX 18, no squeeze
Output: {{"bias":"NEUTRAL","confidence":45,"action":"WAIT","entry":0,"sl":0,"tp1":0,"tp2":0,"reason":"No clear trend, low ADX, wait for breakout"}}
</example>
</examples>

<rules>
- ONLY return valid JSON. No markdown, no explanation outside JSON.
- entry must be at or very near the Current Price above (this bot enters at market; do not propose entries far from the current price)
- entry, sl, tp1, tp2 must be realistic prices (not 0 if action is LONG/SHORT)
- sl must be BELOW entry for LONG, ABOVE entry for SHORT
- tp1 must be ABOVE entry for LONG, BELOW entry for SHORT; tp2 must be beyond tp1
- confidence 0-100 based on confluence of signals (not arbitrary)
- WAIT if confidence < 60 or unclear setup
</rules>

<output_format>
{{"bias":"BULLISH|BEARISH|NEUTRAL","confidence":0-100,"action":"LONG|SHORT|WAIT","entry":number,"sl":number,"tp1":number,"tp2":number,"reason":"concise technical rationale"}}
</output_format>

Return JSON now:"""

    # ---------------- CHATGPT (OpenAI) ----------------
    def _chatgpt_blocking(self, prompt: str) -> AISignal:
        try:
            resp = self.openai_client.chat.completions.create(
                model=self.cfg.chat_model,
                messages=[
                    {"role": "system", "content": "You are an expert crypto quant trader. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=400,
                response_format={"type": "json_object"}
            )
            return self._safe_parse(resp.choices[0].message.content, "chatgpt")
        except Exception as e:
            console.print(f"[red]ChatGPT error: {e}[/red]")
            return AISignal(reason=f"ChatGPT error: {str(e)[:50]}",
                            model=self.cfg.chat_model, ok=False)

    # ---------------- GROK (xAI) ----------------
    def _grok_blocking(self, prompt: str) -> AISignal:
        try:
            resp = self.xai_client.chat.completions.create(
                model=self.cfg.grok_model,
                messages=[
                    {"role": "system", "content": "You are an expert crypto quant trader. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=400,
            )
            return self._safe_parse(resp.choices[0].message.content, "grok")
        except Exception as e:
            console.print(f"[red]Grok error: {e}[/red]")
            return AISignal(reason=f"Grok error: {str(e)[:50]}",
                            model=self.cfg.grok_model, ok=False)

    # ---------------- WEIGHTED CONSENSUS ENGINE ----------------
    @staticmethod
    def _wait_dict(regime: str, reason: str) -> dict:
        """Fully-formed WAIT dict so downstream code never KeyErrors."""
        return {
            "bias": "NEUTRAL", "action": "WAIT",
            "entry": 0.0, "sl": 0.0, "tp1": 0.0, "tp2": 0.0,
            "confidence": 0, "regime": regime, "reason": reason,
        }

    def _weighted_merge(self, a: AISignal, b: AISignal, s: SymbolState) -> dict:
        """
        Weighted consensus based on confidence and model reliability.
        Confidence-weighting is used for price levels; the merged *confidence*
        itself is the plain mean of the two.
        """
        regime = self._detect_regime(s)

        # FIX #1 (cont.): handle API failures explicitly instead of letting a
        # dead model masquerade as a NEUTRAL opinion in the disagreement logic.
        if not a.ok and not b.ok:
            return self._wait_dict(regime, "both AI models failed")
        if a.ok != b.ok:
            good = a if a.ok else b
            console.print(f"[yellow]Warning: only {good.model} responded — "
                          f"single-model signal, confidence penalized[/yellow]")
            return {
                "bias": good.bias,
                "action": good.action,
                "entry": round(good.entry, 4),
                "sl": round(good.sl, 4),
                "tp1": round(good.tp1, 4),
                "tp2": round(good.tp2, 4),
                # haircut: no consensus behind this signal
                "confidence": max(0, good.confidence - 15),
                "regime": regime,
                "reason": f"single-model ({good.model}): {good.reason[:100]}",
            }

        # If models disagree on bias, require higher confidence to proceed
        if a.bias != b.bias:
            if max(a.confidence, b.confidence) < 75:
                return self._wait_dict(
                    regime,
                    f"AI disagreement: ChatGPT={a.bias}({a.confidence}) vs "
                    f"Grok={b.bias}({b.confidence})")
            # If one is much more confident, lean on it but penalize confidence
            dominant = a if a.confidence > b.confidence else b
            console.print(f"[yellow]Warning: AI bias disagreement, using higher confidence ({dominant.model})[/yellow]")
            a = dominant
            b = AISignal(
                bias=dominant.bias,
                confidence=max(0, dominant.confidence - 20),
                action=dominant.action,
                entry=dominant.entry,
                sl=dominant.sl,
                tp1=dominant.tp1,
                tp2=dominant.tp2,
                reason="secondary agreement weak",
                model="consensus_fallback"
            )

        # Confidence-based weights for price-level blending
        total_conf = a.confidence + b.confidence
        if total_conf == 0:
            w_a = w_b = 0.5
        else:
            w_a = a.confidence / total_conf
            w_b = b.confidence / total_conf

        # merged confidence = arithmetic mean (the old a*w_a + b*w_b was the
        # contraharmonic mean and inflated the number)
        merged_confidence = (a.confidence + b.confidence) / 2.0

        action = a.action if a.action == b.action else "WAIT"

        # For SL: take the TIGHTER (safer) one.
        def tighter_sl(sl_a, sl_b):
            if action == "LONG":
                return max(sl_a, sl_b)  # higher SL = tighter for longs
            else:
                return min(sl_a, sl_b)  # lower SL = tighter for shorts

        def consensus_tp(tp_a, tp_b, entry):
            tp = tp_a * w_a + tp_b * w_b
            if action == "LONG":
                return max(tp, entry * 1.005)  # minimum 0.5% above entry
            else:
                return min(tp, entry * 0.995)  # minimum 0.5% below entry

        entry = (a.entry * w_a + b.entry * w_b) if (a.entry and b.entry) else (a.entry or b.entry or s.price)
        sl = tighter_sl(a.sl, b.sl) if (a.sl and b.sl) else (a.sl or b.sl or 0)
        tp1 = consensus_tp(a.tp1, b.tp1, entry) if (a.tp1 and b.tp1) else (a.tp1 or b.tp1 or 0)
        tp2 = consensus_tp(a.tp2, b.tp2, entry) if (a.tp2 and b.tp2) else (a.tp2 or b.tp2 or 0)

        # FIX #2: blending two models' levels (and the 0.5% TP floor) can
        # invert the ladder — e.g. tp1 clamped up past tp2 for a LONG, or
        # tp2 ending up inside tp1 for a SHORT. The bot would then hit "TP2"
        # before "TP1". Enforce ordering after the blend.
        if action == "LONG" and tp1 and tp2 and tp2 < tp1:
            tp1, tp2 = tp2, tp1
        elif action == "SHORT" and tp1 and tp2 and tp2 > tp1:
            tp1, tp2 = tp2, tp1

        return {
            "bias": a.bias,
            "action": action,
            "entry": round(entry, 4),
            "sl": round(sl, 4),
            "tp1": round(tp1, 4),
            "tp2": round(tp2, 4),
            "confidence": round(merged_confidence, 1),
            "reason": f"Consensus: {a.model}({a.confidence})+{b.model}({b.confidence}) | {a.reason[:60]} | {b.reason[:60]}",
            "regime": regime
        }

    # ---------------- ADAPTIVE RISK FILTER ----------------
    def _risk_ok(self, s: SymbolState, signal: dict) -> tuple[bool, str]:
        """Adaptive risk filter with regime-aware thresholds."""

        regime = signal.get("regime", "TRENDING")
        thresholds = self._regime_adjusted_thresholds(regime)

        # The adaptive threshold acts as a FLOOR on the regime's confidence
        # requirement, so the win-rate feedback loop has a real effect.
        min_confidence = max(thresholds["min_confidence"], self.adaptive_conf_threshold)

        # Confidence check
        if signal["confidence"] < min_confidence:
            return False, f"confidence {signal['confidence']} < {min_confidence} ({regime})"

        # ADX check (trend strength)
        if s.adx < thresholds["min_adx"]:
            return False, f"ADX {s.adx:.1f} < {thresholds['min_adx']} ({regime})"

        # Volume check
        if not s.volume_ok:
            return False, "volume insufficient"

        # Action check
        if signal["action"] == "WAIT":
            return False, "action is WAIT"

        # Entry validation
        if signal["entry"] <= 0:
            return False, "invalid entry price"

        # FIX #3: entry must be near the live price. The bot enters at MARKET,
        # so an AI "entry" 2% away (a pullback fantasy) makes every downstream
        # SL/TP/R:R number wrong for the actual fill. Reject drifted entries.
        if s.price > 0:
            max_drift = max(s.atr * 1.0, s.price * 0.003) if s.atr > 0 else s.price * 0.005
            if abs(signal["entry"] - s.price) > max_drift:
                return False, (f"entry {signal['entry']} too far from market "
                               f"{s.price} (>{max_drift:.4f})")

        # Stop loss validation
        if signal["sl"] <= 0:
            return False, "invalid stop loss"

        # Validate SL direction
        if signal["action"] == "LONG" and signal["sl"] >= signal["entry"]:
            return False, f"LONG SL {signal['sl']} >= entry {signal['entry']}"
        if signal["action"] == "SHORT" and signal["sl"] <= signal["entry"]:
            return False, f"SHORT SL {signal['sl']} <= entry {signal['entry']}"

        # FIX #4: validate the take-profits too. Before, tp1=0 (models omitted
        # it) made reward = |0 - entry| = the entire entry price, so R:R came
        # out astronomically high and the filter happily passed a signal with
        # NO take-profit. A wrong-side tp1 also slipped through.
        if signal["tp1"] <= 0:
            return False, "missing tp1"
        if signal["action"] == "LONG" and signal["tp1"] <= signal["entry"]:
            return False, f"LONG tp1 {signal['tp1']} <= entry {signal['entry']}"
        if signal["action"] == "SHORT" and signal["tp1"] >= signal["entry"]:
            return False, f"SHORT tp1 {signal['tp1']} >= entry {signal['entry']}"
        if signal["tp2"]:
            if signal["action"] == "LONG" and signal["tp2"] <= signal["tp1"]:
                return False, f"LONG tp2 {signal['tp2']} <= tp1 {signal['tp1']}"
            if signal["action"] == "SHORT" and signal["tp2"] >= signal["tp1"]:
                return False, f"SHORT tp2 {signal['tp2']} >= tp1 {signal['tp1']}"

        # Risk/Reward calculation
        risk = abs(signal["entry"] - signal["sl"])
        reward1 = abs(signal["tp1"] - signal["entry"])

        if risk == 0:
            return False, "zero risk distance"

        rr = reward1 / risk

        if rr < thresholds["min_rr"]:
            return False, f"R:R {rr:.2f} < {thresholds['min_rr']} ({regime})"

        # ATR-based sanity check: SL shouldn't be more than 3x ATR away
        if s.atr > 0 and risk > 3 * s.atr:
            return False, f"SL too wide: {risk:.4f} > 3x ATR ({3*s.atr:.4f})"

        # Position sizing sanity: minimum move should be > 0.2% (avoid noise)
        min_move_pct = risk / signal["entry"]
        if min_move_pct < 0.002:  # 0.2%
            return False, f"stop too tight: {min_move_pct*100:.2f}% < 0.2%"

        return True, f"passed ({regime}, R:R {rr:.2f})"

    # ---------------- MAIN DUAL AI ANALYSIS ----------------
    async def deep_analysis(self, s: SymbolState) -> dict:
        loop = asyncio.get_running_loop()
        prompt = self._build_prompt(s)

        # Run both AIs in parallel
        chat_task = loop.run_in_executor(self.executor, self._chatgpt_blocking, prompt)
        grok_task = loop.run_in_executor(self.executor, self._grok_blocking, prompt)

        chatgpt_signal, grok_signal = await asyncio.gather(chat_task, grok_task)

        # Log individual signals for tracking
        console.print(f"[dim]ChatGPT: {chatgpt_signal.bias} {chatgpt_signal.action} conf={chatgpt_signal.confidence}[/dim]")
        console.print(f"[dim]Grok: {grok_signal.bias} {grok_signal.action} conf={grok_signal.confidence}[/dim]")

        # Merge with weighted consensus
        signal = self._weighted_merge(chatgpt_signal, grok_signal, s)

        # Apply adaptive risk filter
        passed, risk_reason = self._risk_ok(s, signal)

        if not passed:
            # FIX #5: return a fully-formed dict (bias/regime/levels included)
            # so compute_signal & friends read consistent keys either way.
            return {
                **self._wait_dict(signal.get("regime", "TRENDING"),
                                  f"Risk filter: {risk_reason}"),
                "bias": signal.get("bias", "NEUTRAL"),
                "raw_signal": signal,
                "chatgpt": chatgpt_signal.to_dict(),
                "grok": grok_signal.to_dict()
            }

        # Add metadata for downstream tracking
        signal["passed_filter"] = True
        signal["filter_reason"] = risk_reason
        signal["symbol"] = s.symbol
        signal["chatgpt"] = chatgpt_signal.to_dict()
        signal["grok"] = grok_signal.to_dict()

        # FIX #6: use wall-clock time. loop.time() is a monotonic clock with an
        # arbitrary epoch — mixing it with time.time() (used in
        # update_signal_result) made the timestamps in signal_history
        # incomparable.
        self.signal_history.append({
            "timestamp": time.time(),
            "symbol": s.symbol,
            "signal": signal.copy(),
            "price": s.price
        })

        # Trim history
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-500:]

        return signal

    # ---------------- SENTIMENT WITH CONTEXT ----------------
    def _sentiment_blocking(self, symbol: str) -> str:
        # FIX #7: the old prompt asked the model to "consider recent price
        # action, funding rates, social sentiment" — data a plain chat
        # completion does NOT have. It answered from training-set vibes and
        # dressed hallucination up as sentiment. Now the model is told to
        # default to NEUTRAL unless it has a genuinely strong prior, which is
        # the honest (and safe) behavior: NEUTRAL doesn't block any trade.
        # For real sentiment, feed funding rate / OI / long-short ratio from
        # the exchange into this prompt instead.
        try:
            resp = self.openai_client.chat.completions.create(
                model=self.cfg.chat_model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"What is the general structural market sentiment for {symbol} "
                        f"crypto futures based on what you reliably know about this asset? "
                        f"You have no live data — if you cannot justify a strong directional "
                        f"view without current information, answer NEUTRAL. "
                        f"Return exactly one word: BULLISH, BEARISH, or NEUTRAL."
                    )
                }],
                max_completion_tokens=10,
            )
            sent = (resp.choices[0].message.content or "").strip().upper()
            return sent if sent in {"BULLISH", "BEARISH", "NEUTRAL"} else "NEUTRAL"
        except Exception:
            return "NEUTRAL"

    async def sentiment(self, symbol: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._sentiment_blocking, symbol)

    # ---------------- PERFORMANCE TRACKING ----------------
    def update_signal_result(self, entry_price: float, exit_price: float, signal: dict):
        """Call this after a trade closes to track AI accuracy and adapt thresholds."""
        if not signal or signal.get("action") == "WAIT":
            return

        pnl = exit_price - entry_price if signal["action"] == "LONG" else entry_price - exit_price
        pnl_pct = (pnl / entry_price * 100) if entry_price else 0.0

        # Persist the result so the feedback loop below sees closed trades.
        self.signal_history.append({
            "timestamp": time.time(),
            "symbol": signal.get("symbol", "?"),
            "signal": signal,
            "entry": entry_price,
            "exit": exit_price,
            "result": pnl,
            "pnl_pct": pnl_pct,
        })

        # Use `"result" in h`, not `h.get("result")` — a losing or break-even
        # trade has 0/negative PnL, which is falsy and would be dropped.
        #
        # FIX #8: adapt on a SLIDING WINDOW of the last 30 closed trades, not
        # the all-time list. With the all-time stats, 200 old trades meant a
        # bad recent streak barely moved the win rate and the threshold
        # effectively froze; a rolling window actually reacts to current
        # performance.
        recent_results = [h for h in self.signal_history if "result" in h][-30:]
        wins = sum(1 for h in recent_results if h["result"] > 0)
        total = len(recent_results)

        if total >= 20:
            win_rate = wins / total
            # Adjust confidence threshold based on performance
            if win_rate < 0.45:
                self.adaptive_conf_threshold = min(90, self.adaptive_conf_threshold + 2)
                console.print(f"[yellow]Win rate low ({win_rate:.1%}), raising conf threshold to {self.adaptive_conf_threshold}[/yellow]")
            elif win_rate > 0.60:
                self.adaptive_conf_threshold = max(65, self.adaptive_conf_threshold - 1)
                console.print(f"[green]Win rate good ({win_rate:.1%}), lowering conf threshold to {self.adaptive_conf_threshold}[/green]")