import os

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

import asyncio
import random
import time
import pandas as pd
import indicators as ta
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from config import BotConfig
from position import Position
from symbol_state import SymbolState
from trader_logger import TradeLogger
from ai_analysis import AIAnalyst
from exchange import Exchange
from dashboard import Dashboard
from noop_live import _NoopLive
from round_step import round_step
from telegram_notification import Telegram
from binance import BinanceSocketManager
from binance.enums import (
    SIDE_BUY, SIDE_SELL
)
console = Console()

# NOTE: make sure position.py's Position dataclass has these fields with
# defaults, otherwise the runner logic will raise AttributeError:
#   runner_active: bool = False
#   runner_qty: float = 0.0


class TradingBot:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.headless = cfg.resolve_headless()

        self.exchange = Exchange(cfg)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.ai = AIAnalyst(cfg, self.executor)
        self.telegram = Telegram(cfg)
        self.logger = TradeLogger(cfg.log_file)
        self.dashboard = Dashboard(self)

        # ── state ───────────────────────────────────────────────────────────
        self.symbol_pool: list[str] = list(cfg.symbol_pool)
        self.symbol_list: list[str] = random.sample(
            self.symbol_pool, min(cfg.active_symbol_count, len(self.symbol_pool)))
        self.symbols: dict[str, SymbolState] = {
            s: SymbolState(symbol=s) for s in self.symbol_list}

        self.trade_log: list[dict] = []
        self.session_start_balance: float = 0.0
        self.daily_realized_pnl: float = 0.0
        self.trading_halted: bool = False

        self.last_reshuffle: float = time.time()
        self.ws_needs_restart: bool = False
        self._indicator_index: int = 0
        self._last_indicator_tick: float = 0.0
        self._bg_tasks: set[asyncio.Task] = set()
        self._ai_inflight: set[str] = set()

        # Performance tracking for adaptive thresholds
        self._ai_performance: dict[str, list[float]] = {}  # symbol -> [pnl% list]
        self._last_ai_trade: dict[str, dict] = {}  # symbol -> {entry, signal, ts}

    # ── convenience accessors ─────────────────────────────────────────────
    @property
    def main_symbol(self) -> str:
        return self.symbol_list[0] if self.symbol_list else ""

    def iter_states(self):
        for sym in self.symbol_list:
            yield self.symbols[sym]

    def ensure_symbol_state(self, sym: str):
        if sym not in self.symbols:
            self.symbols[sym] = SymbolState(symbol=sym)

    def count_open_longs(self) -> int:
        return sum(1 for s in self.iter_states() if s.position.side == "LONG")

    def count_open_shorts(self) -> int:
        return sum(1 for s in self.iter_states() if s.position.side == "SHORT")

    def _correlation_risk_ok(self, sym: str, side: str) -> bool:
        """Prevent stacking multiple positions in same direction on correlated assets."""
        if not hasattr(self.cfg, 'max_correlated_positions'):
            return True
        same_side_count = self.count_open_longs() if side == "LONG" else self.count_open_shorts()
        return same_side_count < getattr(self.cfg, 'max_correlated_positions', 3)

    # ── background AI (never blocks the trading loop) ──────────────────────
    def _spawn(self, coro):
        """Fire-and-forget with strong ref and automatic cleanup."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                coro.close()
            except Exception:
                pass
            return
        task = loop.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _refresh_sentiment(self, sym: str):
        try:
            self.symbols[sym].sentiment = await self.ai.sentiment(sym)
        except Exception as e:
            console.print(f"[yellow]⚠ sentiment {sym}: {type(e).__name__}[/yellow]")
        finally:
            self._ai_inflight.discard(sym)

    async def _refresh_analysis(self, sym: str):
        try:
            analysis = await self.ai.deep_analysis(self.symbols[sym])
            self.symbols[sym].analysis = analysis

            if analysis.get("action") in ("LONG", "SHORT"):
                self._last_ai_trade[sym] = {
                    "entry": self.symbols[sym].price,
                    "signal": analysis,
                    "ts": time.time()
                }

        except Exception as e:
            console.print(f"[yellow]⚠ analysis {sym}: {type(e).__name__}[/yellow]")
        finally:
            self._ai_inflight.discard("A:" + sym)

    # FIX #6: keep the exchange hard stop trailing behind the software trail.
    # Previously the exchange SL never moved once the trail activated, so if
    # the bot crashed the position was still protected only at the original
    # (much wider) stop. This tightens the real order as the trail ratchets.
    async def _sync_hard_stop(self, sym: str):
        try:
            s = self.symbols[sym]
            pos = s.position
            if pos.side is None or not pos.trail_active:
                return
            info = await self.exchange.symbol_info(sym)
            new_sl = round_step(pos.trail_stop, info["tick_size"])
            # only ever tighten the stop, never loosen it
            if pos.side == "LONG" and new_sl <= pos.sl_price:
                return
            if pos.side == "SHORT" and 0 < pos.sl_price <= new_sl:
                return
            pos.sl_order_id = await self.exchange.replace_stop_loss(sym, pos.side, new_sl)
            pos.sl_price = new_sl
        except Exception as e:
            console.print(f"[yellow]⚠ hard-stop sync {sym}: {type(e).__name__}[/yellow]")
        finally:
            self._ai_inflight.discard("SL:" + sym)

    # ── symbol management ──────────────────────────────────────────────────
    async def filter_tradable_symbols(self):
        live_pool, new_active, dead = await self.exchange.filter_tradable(
            self.symbol_pool, self.symbol_list, self.cfg.active_symbol_count)
        if dead:
            console.print(f"[yellow]⚠ Dropping {len(dead)} delisted symbol(s): "
                          f"{', '.join(dead)}[/yellow]")
        self.symbol_pool = live_pool
        self.symbol_list = new_active
        for sym in self.symbol_list:
            self.ensure_symbol_state(sym)
        console.print(f"[green]✅ {len(live_pool)} tradable symbols verified; "
                      f"{len(self.symbol_list)} active.[/green]")

    def reshuffle_symbols(self):
        """Re-pick random active set; symbols with open positions are pinned."""
        pinned = [s for s in self.symbol_list if self.symbols[s].position.is_open]
        pool_rest = [s for s in self.symbol_pool if s not in pinned]
        random.shuffle(pool_rest)
        slots = max(0, self.cfg.active_symbol_count - len(pinned))
        new_symbols = pinned + pool_rest[:slots]
        random.shuffle(new_symbols)

        for sym in new_symbols:
            self.ensure_symbol_state(sym)

        removed = [s for s in self.symbol_list if s not in new_symbols]
        added = [s for s in new_symbols if s not in self.symbol_list]
        self.symbol_list = new_symbols
        self._indicator_index = 0
        if added or removed:
            self.ws_needs_restart = True
            console.print(
                f"[magenta]🔀 Symbol reshuffle | +{len(added)} {added[:3]}"
                f"{'…' if len(added) > 3 else ''} | -{len(removed)} {removed[:3]}"
                f"{'…' if len(removed) > 3 else ''}[/magenta]"
            )
            self.telegram.notify(f"🔀 Symbol reshuffle\n+{added}\n-{removed}", "INFO")

    # ── leverage ───────────────────────────────────────────────────────────
    async def setup_leverage(self):
        for sym in self.symbol_list:
            await self.exchange.ensure_leverage(sym)

    # ── position sync / adoption ──────────────────────────────────────────
    async def _adopt_position(self, sym: str, side: str, qty: float,
                              entry: float, unrealized: float, pnl_pct: float):
        s = self.symbols[sym]
        atr = s.atr if s.atr > 0 else entry * 0.005
        sl_d, tp1_d, tp2_d = (atr * self.cfg.atr_sl_mult,
                              atr * self.cfg.atr_tp1_mult, atr * self.cfg.atr_tp2_mult)
        if side == "LONG":
            sl_p, tp1_p, tp2_p = entry - sl_d, entry + tp1_d, entry + tp2_d
        else:
            sl_p, tp1_p, tp2_p = entry + sl_d, entry - tp1_d, entry - tp2_d

        info = await self.exchange.symbol_info(sym)
        sl_p = round_step(sl_p, info["tick_size"])
        tp1_p = round_step(tp1_p, info["tick_size"])
        tp2_p = round_step(tp2_p, info["tick_size"])

        await self.exchange.cancel_open_orders(sym)
        sl_id = await self.exchange.place_stop_loss(sym, side, sl_p)

        s.position = Position(
            side=side, qty=qty, entry=entry, sl_price=sl_p,
            tp1_price=tp1_p, tp2_price=tp2_p, trail_best=entry, trail_stop=sl_p,
            pnl_pct=pnl_pct, pnl_usdt=unrealized, sl_order_id=sl_id, open_time=time.time(),
        )
        console.print(f"[cyan]🧬 Adopted {side} {sym} @ ${entry:,.4f} | "
                      f"SL ${sl_p:,.4f}  TP1 ${tp1_p:,.4f}  TP2 ${tp2_p:,.4f}[/cyan]")
        self.telegram.notify(
            f"🧬 Adopted existing {side} {sym} @ ${entry:,.4f}\n"
            f"SL ${sl_p:,.4f} | TP1 ${tp1_p:,.4f} | TP2 ${tp2_p:,.4f}", "TRADE")

    async def sync_positions(self):
        try:
            for p in await self.exchange.positions_info():
                sym = p["symbol"]
                if sym not in self.symbols:
                    continue
                s = self.symbols[sym]
                pos = s.position
                qty = float(p.get("positionAmt", 0))
                now = time.time()

                if qty == 0:
                    # Position was closed on the exchange (hard SL hit, manual
                    # close, liquidation…). Evaluate AI, then reset.
                    if pos.side is not None and now - pos.open_time >= self.cfg.position_adopt_grace:
                        self._evaluate_ai_performance(sym, s.price, pos)
                        s.reset_position()
                    continue

                entry = float(p.get("entryPrice", 0))
                unrealized = float(p.get("unRealizedProfit", 0))
                pnl_pct = (unrealized / (abs(qty) * entry) * 100) if entry > 0 else 0.0
                side = "LONG" if qty > 0 else "SHORT"

                untracked = (pos.side is None or pos.side != side or pos.sl_price <= 0)
                if untracked:
                    if now - pos.open_time < self.cfg.position_adopt_grace:
                        pos.qty, pos.entry, pos.pnl_pct, pos.pnl_usdt = abs(qty), entry, pnl_pct, unrealized
                        continue
                    await self._adopt_position(sym, side, abs(qty), entry, unrealized, pnl_pct)
                else:
                    pos.qty, pos.entry, pos.pnl_pct, pos.pnl_usdt = abs(qty), entry, pnl_pct, unrealized
        except Exception as e:
            console.print(f"[yellow]⚠ sync_positions: {e}[/yellow]")

    def _evaluate_ai_performance(self, sym: str, exit_price: float, pos: Position):
        if sym not in self._last_ai_trade:
            return
        ai_record = self._last_ai_trade[sym]
        if ai_record.get("signal", {}).get("action") != pos.side:
            return  # AI didn't signal this direction

        entry = ai_record["entry"]
        if entry <= 0 or exit_price <= 0:
            return
        pnl_pct = ((exit_price - entry) / entry * 100) if pos.side == "LONG" else ((entry - exit_price) / entry * 100)

        if sym not in self._ai_performance:
            self._ai_performance[sym] = []
        self._ai_performance[sym].append(pnl_pct)
        self._ai_performance[sym] = self._ai_performance[sym][-50:]  # keep last 50

        try:
            self.ai.update_signal_result(entry, exit_price, ai_record["signal"])
        except Exception:
            pass

    # ── open / close ───────────────────────────────────────────────────────
    async def open_position(self, sym: str, side: str, balance: float, ai_signal: dict = None) -> bool:
        await self.exchange.ensure_leverage(sym)
        s = self.symbols[sym]
        cfg = self.cfg
        price = s.price
        if price <= 0:
            return False
        atr = s.atr if s.atr > 0 else price * 0.005
        info = await self.exchange.symbol_info(sym)

        # FIX #3: a market order fills at the *market* price, not at the AI's
        # suggested entry. Previously Position.entry was set to the AI entry,
        # which skewed PnL, breakeven and trailing maths, and could even put
        # the SL on the wrong side of the actual fill. We now always anchor
        # the position at the live price and only borrow AI SL/TP levels when
        # they are valid relative to that price.
        #
        # FIX #4: the old fallbacks (entry + atr * mult) pointed the wrong way
        # for SHORTs, and tp2 was never validated — a SHORT could end up with
        # tp2 ABOVE entry, making `p <= tp2` true instantly → immediate bogus
        # "TP2" close right after opening. Fallbacks are now direction-aware
        # and every level is validated.
        entry = price
        sl_p = tp1_p = tp2_p = None
        if ai_signal:
            ai_sl = float(ai_signal.get("sl") or 0)
            ai_tp1 = float(ai_signal.get("tp1") or 0)
            ai_tp2 = float(ai_signal.get("tp2") or 0)
            if side == "LONG":
                ok = (0 < ai_sl < price
                      and (ai_tp1 == 0 or ai_tp1 > price)
                      and (ai_tp2 == 0 or ai_tp2 > max(price, ai_tp1)))
                if ok:
                    sl_p = ai_sl
                    tp1_p = ai_tp1 if ai_tp1 else price + atr * cfg.atr_tp1_mult
                    tp2_p = ai_tp2 if ai_tp2 else max(tp1_p, price + atr * cfg.atr_tp2_mult)
            else:  # SHORT
                ok = (ai_sl > price
                      and (ai_tp1 == 0 or 0 < ai_tp1 < price)
                      and (ai_tp2 == 0 or 0 < ai_tp2 < (ai_tp1 if ai_tp1 else price)))
                if ok:
                    sl_p = ai_sl
                    tp1_p = ai_tp1 if ai_tp1 else price - atr * cfg.atr_tp1_mult
                    tp2_p = ai_tp2 if ai_tp2 else min(tp1_p, price - atr * cfg.atr_tp2_mult)
            # sanity: reject stops that are absurdly wide or razor-thin
            if sl_p is not None:
                d = abs(price - sl_p)
                if not (atr * 0.2 <= d <= atr * 4.0):
                    sl_p = tp1_p = tp2_p = None

        if sl_p is None:  # ATR-based defaults
            sl_d = atr * cfg.atr_sl_mult
            tp1_d, tp2_d = atr * cfg.atr_tp1_mult, atr * cfg.atr_tp2_mult
            sl_p = price - sl_d if side == "LONG" else price + sl_d
            tp1_p = price + tp1_d if side == "LONG" else price - tp1_d
            tp2_p = price + tp2_d if side == "LONG" else price - tp2_d

        # FIX #5: true fixed-fractional sizing. qty = risk / stop-distance
        # already loses exactly `risk_percent` of balance when the stop hits.
        # The old `* cfg.leverage` multiplier made every stop-out cost
        # leverage × risk_percent (e.g. 10x leverage at 1% risk = -10% per
        # loss). Leverage now only reduces margin used, as it should.
        # If you truly want the old aggressive sizing back, re-add the
        # multiplier consciously.
        risk_usdt = balance * cfg.risk_percent
        sl_distance = abs(entry - sl_p)
        if sl_distance <= 0:
            sl_distance = atr * 0.5  # safety fallback
        raw_qty = risk_usdt / sl_distance

        qty = round_step(raw_qty, info["step_size"])

        avail = await self.exchange.available_margin()
        max_qty_margin = (avail * cfg.margin_safety * cfg.leverage) / price if price > 0 else 0.0
        if max_qty_margin < qty:
            qty = round_step(max_qty_margin, info["step_size"])
        qty = max(qty, info["min_qty"])
        required_margin = qty * price / cfg.leverage

        if qty * price < info["min_notional"] or required_margin > avail * cfg.margin_safety:
            console.print(f"[yellow]⚠ {sym} can't size order — need ${required_margin:,.2f} "
                          f"margin, have ${avail:,.2f}; cooling down[/yellow]")
            s.cooldown_until = time.time() + cfg.cooldown_after_loss
            return False

        order_side = SIDE_BUY if side == "LONG" else SIDE_SELL
        if not await self.exchange.market_order(sym, order_side, qty):
            s.cooldown_until = time.time() + cfg.cooldown_after_loss
            return False

        sl_p = round_step(sl_p, info["tick_size"])
        tp1_p = round_step(tp1_p, info["tick_size"])
        tp2_p = round_step(tp2_p, info["tick_size"])

        sl_id = await self.exchange.place_stop_loss(sym, side, sl_p)
        s.position = Position(
            side=side, qty=qty, entry=entry, sl_price=sl_p, tp1_price=tp1_p, tp2_price=tp2_p,
            trail_best=entry, trail_stop=sl_p, sl_order_id=sl_id, open_time=time.time(),
        )
        self.exchange.invalidate_balance_cache()

        row = {"date": date.today().isoformat(), "time": datetime.now().strftime("%H:%M:%S"),
               "symbol": sym, "action": "OPEN", "side": side, "entry": entry, "exit": "",
               "qty": qty, "sl": sl_p, "tp1": tp1_p, "tp2": tp2_p,
               "pnl_pct": "", "pnl_usdt": "", "reason": "AI_SIGNAL" if ai_signal else "SIGNAL"}
        self.logger.log(row)
        self.trade_log.append({**row, "price": price})
        self.trade_log[:] = self.trade_log[-20:]

        self.telegram.notify(
            f"🟢 OPEN {side} {sym} @ ${entry:,.4f}\n"
            f"qty {qty} | SL ${sl_p:,.4f} | TP1 ${tp1_p:,.4f} | TP2 ${tp2_p:,.4f}", "TRADE")
        return True

    async def close_position(self, sym: str, reason: str = "SIGNAL",
                             partial: bool = False, partial_qty: float = 0.0,
                             move_sl_to_breakeven: bool = True):
        s = self.symbols[sym]
        pos = s.position
        side = pos.side
        qty = partial_qty if partial else pos.qty
        if side is None or qty == 0:
            return

        if not partial:
            await self.exchange.cancel_open_orders(sym)
        close_side = SIDE_SELL if side == "LONG" else SIDE_BUY
        if not await self.exchange.market_order(sym, close_side, qty, reduce_only=True):
            return

        # FIX #7: realize PnL from the live price instead of the last
        # position-sync snapshot (which can be up to position_sync_interval
        # seconds stale). Keeps daily_realized_pnl and the loss-limit halt
        # honest on fast moves.
        if pos.entry > 0 and s.price > 0:
            diff = (s.price - pos.entry) if side == "LONG" else (pos.entry - s.price)
            pnl_usdt = diff * qty
        else:
            pnl_usdt = pos.pnl_usdt * (qty / pos.qty) if pos.qty > 0 else 0.0
        self.daily_realized_pnl += pnl_usdt
        self.exchange.invalidate_balance_cache()

        pnl_col = "green" if pnl_usdt >= 0 else "red"
        console.print(f"[cyan]🔒 {'PARTIAL ' if partial else ''}Close {side} {sym} | "
                      f"{reason} | PnL: [{pnl_col}]{pos.pnl_pct:.2f}%[/][/cyan]")

        row = {"date": date.today().isoformat(), "time": datetime.now().strftime("%H:%M:%S"),
               "symbol": sym, "action": "PARTIAL_CLOSE" if partial else "CLOSE", "side": side,
               "entry": pos.entry, "exit": s.price, "qty": qty, "sl": pos.sl_price,
               "tp1": pos.tp1_price, "tp2": pos.tp2_price, "pnl_pct": f"{pos.pnl_pct:.2f}%",
               "pnl_usdt": f"{pnl_usdt:.2f}", "reason": reason}
        self.logger.log(row)
        self.trade_log.append({**row, "price": s.price, "pnl": f"{pos.pnl_pct:+.2f}%"})
        self.trade_log[:] = self.trade_log[-20:]

        kind = "🟠 PARTIAL CLOSE" if partial else "🔴 CLOSE"
        self.telegram.notify(
            f"{kind} {side} {sym} @ ${s.price:,.4f}\n"
            f"{reason} | PnL {pos.pnl_pct:+.2f}% (${pnl_usdt:+.2f})", "TRADE")

        if partial:
            pos.qty -= qty
            pos.tp1_hit = True
            # FIX #2: only move the SL to breakeven for TP1 partials. The TP2
            # runner exit manages its own (tighter) trailing stop — the old
            # code stomped it back to breakeven right after setting the trail.
            if move_sl_to_breakeven:
                buffer = abs(pos.entry - pos.sl_price) * 0.1 if pos.sl_price != pos.entry else s.atr * 0.2
                if side == "LONG":
                    pos.sl_price = pos.entry + buffer
                else:
                    pos.sl_price = pos.entry - buffer
                pos.sl_order_id = await self.exchange.replace_stop_loss(sym, side, pos.sl_price)
                console.print(f"[cyan]  → SL moved to breakeven+ @ ${pos.sl_price:,.4f}[/cyan]")
        else:
            if pnl_usdt < 0:
                s.cooldown_until = time.time() + self.cfg.cooldown_after_loss
                console.print(f"[yellow]⏳ {sym} cooldown {self.cfg.cooldown_after_loss}s "
                              f"after loss[/yellow]")
            self._evaluate_ai_performance(sym, s.price, pos)
            s.reset_position()

    # FIX #1 (the big one): TP2 / runner exit, deduplicated for both sides.
    # The old code re-triggered on EVERY tick because after the first
    # TP2-PARTIAL the price was still beyond tp2_price and nothing marked the
    # runner as active in the entry condition. Result: the "runner" got
    # chipped away partial-by-partial each ~2s tick until it was fully closed
    # — it never actually ran. The caller now guards on `not pos.runner_active`
    # and this helper arms a tight trail + a real exchange stop behind the
    # remaining quantity.
    async def _tp2_exit(self, sym: str):
        s = self.symbols[sym]
        pos = s.position
        side = pos.side
        p = s.price
        cfg = self.cfg
        info = await self.exchange.symbol_info(sym)
        close_qty = round_step(pos.qty * cfg.runner_fraction, info["step_size"])

        if close_qty >= info["min_qty"] and pos.qty - close_qty >= info["min_qty"]:
            await self.close_position(sym, reason="TP2-PARTIAL", partial=True,
                                      partial_qty=close_qty, move_sl_to_breakeven=False)
            if pos.side is None:  # safety — fully closed elsewhere
                return
            pos.runner_active = True
            pos.runner_qty = pos.qty  # remaining qty (already deducted)
            pos.trail_active = True
            pos.trail_best = p  # ratchet from here, not from stale trail_best
            atr = s.atr if s.atr > 0 else pos.entry * 0.005
            trail_offset = atr * cfg.atr_sl_mult * cfg.runner_trail_mult
            if side == "LONG":
                pos.trail_stop = max(pos.trail_stop, p - trail_offset)
            else:
                pos.trail_stop = min(pos.trail_stop, p + trail_offset)
            # park a real exchange stop behind the runner (crash protection)
            hard = round_step(pos.trail_stop, info["tick_size"])
            try:
                pos.sl_order_id = await self.exchange.replace_stop_loss(sym, side, hard)
                pos.sl_price = hard
            except Exception as e:
                console.print(f"[yellow]⚠ runner hard-stop {sym}: {type(e).__name__}[/yellow]")
            console.print(f"[green]🚀 {sym} runner: {pos.qty} qty @ trail "
                          f"${pos.trail_stop:,.4f}[/green]")
        else:
            # position too small to split — take the full TP2
            await self.close_position(sym, reason="TP2")

    # ── indicators ───────────────────────────────────────────────────────
    def _bb_squeeze(self, df: pd.DataFrame) -> bool:
        try:
            bb = ta.bbands(df["Close"], length=self.cfg.squeeze_lookback, std=self.cfg.bb_std)
            if bb is None or bb.empty:
                return False
            bbb_cols = [c for c in bb.columns if c.upper().startswith("BBB")]
            if bbb_cols:
                width = bb[bbb_cols[0]]
            else:
                upper = next((c for c in bb.columns if c.upper().startswith("BBU")), None)
                lower = next((c for c in bb.columns if c.upper().startswith("BBL")), None)
                mid = next((c for c in bb.columns if c.upper().startswith("BBM")), None)
                if not (upper and lower and mid):
                    return False
                width = (bb[upper] - bb[lower]) / bb[mid].replace(0, pd.NA)
            if width.dropna().empty:
                return False
            ref = width.rolling(50).quantile(0.20).iloc[-1]
            cur = width.iloc[-1]
            if pd.isna(ref) or pd.isna(cur):
                return False
            return bool(cur < ref)
        except Exception:
            return False

    async def update_one_symbol(self, sym: str):
        df15, df1h = await asyncio.gather(
            self.exchange.fetch_df(sym, self.cfg.tf_primary, 250),
            self.exchange.fetch_df(sym, self.cfg.tf_confirm, 60),
        )
        if df15 is None or len(df15) < 60:
            return

        df15["RSI"] = ta.rsi(df15["Close"], length=14)
        df15["SMA20"] = ta.sma(df15["Close"], length=20)
        df15["EMA50"] = ta.ema(df15["Close"], length=50)
        df15["EMA200"] = ta.ema(df15["Close"], length=200)
        df15["ATR"] = ta.atr(df15["High"], df15["Low"], df15["Close"], length=14)
        adx_df = ta.adx(df15["High"], df15["Low"], df15["Close"], length=14)
        macd_df = ta.macd(df15["Close"], fast=12, slow=26, signal=9)
        vol_avg = df15["Volume"].rolling(20).mean()

        last = df15.iloc[-1]
        s = self.symbols[sym]

        def _f(val, fallback):
            return float(val) if not pd.isna(val) else fallback

        if not s.price_is_fresh(self.cfg.stale_price_threshold):
            s.price = _f(last["Close"], s.price)
            s.price_ts = time.time()

        s.sma20 = _f(last["SMA20"], s.sma20)
        s.ema50 = _f(last["EMA50"], s.ema50)
        s.ema200 = _f(last["EMA200"], s.ema200)
        s.rsi_15 = _f(last["RSI"], s.rsi_15)
        s.atr = _f(last["ATR"], s.atr)
        s.volume_ok = (float(last["Volume"]) > float(vol_avg.iloc[-1])
                       if not pd.isna(vol_avg.iloc[-1]) else False)
        s.ind_ts = time.time()

        if adx_df is not None and "ADX_14" in adx_df.columns:
            s.adx = _f(adx_df["ADX_14"].iloc[-1], s.adx)
        if macd_df is not None:
            s.macd = _f(macd_df["MACD_12_26_9"].iloc[-1], s.macd)
            s.macd_sig = _f(macd_df["MACDs_12_26_9"].iloc[-1], s.macd_sig)
        s.bb_squeeze = self._bb_squeeze(df15)
        if df1h is not None and len(df1h) > 14:
            df1h["RSI1H"] = ta.rsi(df1h["Close"], length=14)
            s.rsi_1h = _f(df1h["RSI1H"].iloc[-1], s.rsi_1h)

    # ── signal logic ───────────────────────────────────────────────────────
    def compute_signal(self, sym: str) -> str:
        s = self.symbols[sym]
        pos = s.position
        p = s.price
        cfg = self.cfg

        if pos.side is None and time.time() < s.cooldown_until:
            s.hold_reason = "cooldown"
            return "HOLD"

        trending = s.adx >= cfg.adx_min
        macd_bull = s.macd > s.macd_sig
        macd_bear = s.macd < s.macd_sig
        vol_ok = s.volume_ok

        ai = s.analysis if hasattr(s, 'analysis') else None
        ai_fresh = ai and (time.time() - getattr(s, 'analysis_ts', 0) < cfg.analysis_interval * 2)
        ai_action = ai.get("action") if ai_fresh else "WAIT"
        ai_bias = ai.get("bias", "NEUTRAL") if ai_fresh else "NEUTRAL"
        ai_conf = ai.get("confidence", 0) if ai_fresh else 0

        if pos.side is None:
            long_ema = p > s.ema50 > s.ema200
            long_rsi = 38 <= s.rsi_15 <= 65 and s.rsi_1h < 70
            long_sent = s.sentiment != "BEARISH"
            long_tech = long_ema and long_rsi and macd_bull and vol_ok and trending and long_sent

            short_ema = p < s.ema50 < s.ema200
            short_rsi = 35 <= s.rsi_15 <= 62 and s.rsi_1h > 30
            short_sent = s.sentiment != "BULLISH"
            short_tech = short_ema and short_rsi and macd_bear and vol_ok and trending and short_sent

            ai_long_agree = ai_action == "LONG" and ai_bias == "BULLISH" and ai_conf >= cfg.ai_min_confidence
            ai_short_agree = ai_action == "SHORT" and ai_bias == "BEARISH" and ai_conf >= cfg.ai_min_confidence
            ai_long_disagree = ai_action == "SHORT" or (ai_bias == "BEARISH" and ai_conf >= 70)
            ai_short_disagree = ai_action == "LONG" or (ai_bias == "BULLISH" and ai_conf >= 70)

            if long_tech and (ai_long_agree or not ai_long_disagree or not ai_fresh):
                s.hold_reason = f"AI+Tech agree ({ai_conf}%)" if ai_long_agree else ""
                return "BUY"

            if short_tech and (ai_short_agree or not ai_short_disagree or not ai_fresh):
                s.hold_reason = f"AI+Tech agree ({ai_conf}%)" if ai_short_agree else ""
                return "SHORT"

            if short_ema and not long_ema:
                reasons = []
                if not short_rsi: reasons.append("rsi")
                if not macd_bear: reasons.append("macd")
                if not vol_ok: reasons.append("vol")
                if not trending: reasons.append(f"adx{s.adx:.0f}")
                if not short_sent: reasons.append("sent")
                if ai_short_disagree and ai_fresh: reasons.append(f"AI({ai_bias})")
                s.hold_reason = "|".join(reasons) if reasons else "ema"
            else:
                reasons = []
                if not long_ema: reasons.append("ema")
                if not long_rsi: reasons.append("rsi")
                if not macd_bull: reasons.append("macd")
                if not vol_ok: reasons.append("vol")
                if not trending: reasons.append(f"adx{s.adx:.0f}")
                if not long_sent: reasons.append("sent")
                if ai_long_disagree and ai_fresh: reasons.append(f"AI({ai_bias})")
                s.hold_reason = "|".join(reasons) if reasons else "—"

        # Position management
        if pos.side == "LONG":
            s.hold_reason = ""
            ai_exit_signal = ai_fresh and ai_action == "SHORT" and ai_conf >= cfg.ai_strong_confidence and ai_bias == "BEARISH"

            # FIX #8: don't dump a healthy runner on a soft signal. Once the
            # runner is active the tight trail is the exit — only a strong AI
            # reversal can override it. Previously `p < ema50` or one MACD
            # cross would instantly kill the runner right after TP2.
            if pos.runner_active:
                if ai_exit_signal:
                    return "SELL_LONG"
                return "HOLD"

            if (p < s.ema50 or s.rsi_15 > 75 or macd_bear or
                    (s.sentiment == "BEARISH" and p < s.sma20) or ai_exit_signal):
                return "SELL_LONG"

        if pos.side == "SHORT":
            s.hold_reason = ""
            ai_exit_signal = ai_fresh and ai_action == "LONG" and ai_conf >= cfg.ai_strong_confidence and ai_bias == "BULLISH"

            if pos.runner_active:
                if ai_exit_signal:
                    return "COVER_SHORT"
                return "HOLD"

            if (p > s.ema50 or s.rsi_15 < 25 or macd_bull or
                    (s.sentiment == "BULLISH" and p > s.sma20) or ai_exit_signal):
                return "COVER_SHORT"

        return "HOLD"

    async def run_trading_logic(self, balance: float):
        cfg = self.cfg
        if time.time() - self.last_reshuffle >= cfg.symbol_reshuffle_hours * 3600:
            self.reshuffle_symbols()
            self.last_reshuffle = time.time()

        if self.session_start_balance > 0:
            daily_loss_pct = self.daily_realized_pnl / self.session_start_balance
            if daily_loss_pct <= -cfg.daily_loss_limit_pct:
                if not self.trading_halted:
                    console.print(f"[bold red]🚨 Daily loss limit hit "
                                  f"({daily_loss_pct*100:.1f}%). Halting new entries.[/bold red]")
                    self.telegram.notify(f"🚨 Daily loss limit hit ({daily_loss_pct*100:.1f}%). "
                                         f"Halting new entries.", "RISK")
                self.trading_halted = True
            else:
                self.trading_halted = False

        open_longs = self.count_open_longs()
        open_shorts = self.count_open_shorts()

        for sym in list(self.symbol_list):
            s = self.symbols[sym]
            pos = s.position
            p = s.price

            if s.price_ts > 0 and time.time() - s.price_ts > cfg.stale_price_threshold:
                continue

            self._update_trailing_stop(sym)

            # FIX #6: throttled sync of the exchange hard stop with the
            # software trail (only when it has drifted meaningfully).
            if (pos.side is not None and pos.trail_active and pos.sl_price > 0
                    and abs(pos.trail_stop - pos.sl_price) > max(s.atr * 0.5, pos.entry * 0.001)
                    and ("SL:" + sym) not in self._ai_inflight):
                self._ai_inflight.add("SL:" + sym)
                self._spawn(self._sync_hard_stop(sym))

            # ── software SL / TP ──
            if pos.side == "LONG" and pos.entry > 0:
                trail_breach = pos.trail_active and p <= pos.trail_stop
                sl_breach = not pos.trail_active and p <= pos.sl_price
                if trail_breach or sl_breach:
                    await self.close_position(sym, reason="TRAIL-SL" if trail_breach else "SL-SW")
                    open_longs = self.count_open_longs()
                    continue
                if not pos.tp1_hit and p >= pos.tp1_price:
                    info = await self.exchange.symbol_info(sym)
                    half = round_step(pos.qty * cfg.partial_close_fraction, info["step_size"])
                    if 0 < half < pos.qty and half >= info["min_qty"]:
                        await self.close_position(sym, reason="TP1-PARTIAL", partial=True, partial_qty=half)
                    else:
                        pos.tp1_hit = True
                # FIX #1: guard on runner_active so TP2 fires exactly once
                elif p >= pos.tp2_price and not pos.runner_active:
                    await self._tp2_exit(sym)
                    open_longs = self.count_open_longs()
                    continue

            if pos.side == "SHORT" and pos.entry > 0:
                trail_breach = pos.trail_active and p >= pos.trail_stop
                sl_breach = not pos.trail_active and p >= pos.sl_price
                if trail_breach or sl_breach:
                    await self.close_position(sym, reason="TRAIL-SL" if trail_breach else "SL-SW")
                    open_shorts = self.count_open_shorts()
                    continue
                if not pos.tp1_hit and p <= pos.tp1_price:
                    info = await self.exchange.symbol_info(sym)
                    half = round_step(pos.qty * cfg.partial_close_fraction, info["step_size"])
                    if 0 < half < pos.qty and half >= info["min_qty"]:
                        await self.close_position(sym, reason="TP1-PARTIAL", partial=True, partial_qty=half)
                    else:
                        pos.tp1_hit = True
                # FIX #1: same guard for shorts (and the old copy-pasted
                # `if pos.side == "LONG"` dead branch inside the short block
                # is gone — _tp2_exit handles both sides correctly)
                elif p <= pos.tp2_price and not pos.runner_active:
                    await self._tp2_exit(sym)
                    open_shorts = self.count_open_shorts()
                    continue

            signal = self.compute_signal(sym)
            s.signal = signal

            ai = s.analysis if hasattr(s, 'analysis') else None
            ai_fresh = ai and (time.time() - getattr(s, 'analysis_ts', 0) < cfg.analysis_interval * 2)

            # ── signal-flip ──
            flipping = False
            if signal == "BUY" and pos.side == "SHORT":
                console.print(f"[yellow]↩ Signal flip {sym}: closing SHORT → opening LONG[/yellow]")
                await self.close_position(sym, reason="FLIP-TO-LONG")
                open_shorts = self.count_open_shorts()
                pos = s.position
                flipping = True
            elif signal == "SHORT" and pos.side == "LONG":
                console.print(f"[yellow]↩ Signal flip {sym}: closing LONG → opening SHORT[/yellow]")
                await self.close_position(sym, reason="FLIP-TO-SHORT")
                open_longs = self.count_open_longs()
                pos = s.position
                flipping = True
            if flipping:
                s.cooldown_until = 0.0

            # ── open LONG ──
            if signal == "BUY" and pos.side is None:
                if self.trading_halted or balance < cfg.min_usdt_to_trade:
                    s.hold_reason = "halted" if self.trading_halted else "low_balance"
                    continue
                if open_longs >= cfg.max_open_longs:
                    s.hold_reason = f"max_longs({open_longs}/{cfg.max_open_longs})"
                    continue
                if not flipping and time.time() < s.cooldown_until:
                    s.hold_reason = "cooldown"
                    continue
                if not self._correlation_risk_ok(sym, "LONG"):
                    s.hold_reason = "correlation"
                    continue

                ai_signal = ai if (ai_fresh and ai.get("action") == "LONG") else None
                if await self.open_position(sym, "LONG", balance, ai_signal=ai_signal):
                    open_longs += 1
                    s.hold_reason = ""
                else:
                    s.hold_reason = "open_failed"

            # ── open SHORT ──
            elif signal == "SHORT" and pos.side is None:
                if self.trading_halted or balance < cfg.min_usdt_to_trade:
                    s.hold_reason = "halted" if self.trading_halted else "low_balance"
                    continue
                if open_shorts >= cfg.max_open_shorts:
                    s.hold_reason = f"max_shorts({open_shorts}/{cfg.max_open_shorts})"
                    continue
                if not flipping and time.time() < s.cooldown_until:
                    s.hold_reason = "cooldown"
                    continue
                if not self._correlation_risk_ok(sym, "SHORT"):
                    s.hold_reason = "correlation"
                    continue

                ai_signal = ai if (ai_fresh and ai.get("action") == "SHORT") else None
                if await self.open_position(sym, "SHORT", balance, ai_signal=ai_signal):
                    open_shorts += 1
                    s.hold_reason = ""
                else:
                    s.hold_reason = "open_failed"

            # ── non-flip close signals ──
            elif signal == "SELL_LONG" and pos.side == "LONG":
                await self.close_position(sym, reason="SIGNAL")
                open_longs = self.count_open_longs()
            elif signal == "COVER_SHORT" and pos.side == "SHORT":
                await self.close_position(sym, reason="SIGNAL")
                open_shorts = self.count_open_shorts()

    def _update_trailing_stop(self, sym: str):
        pos = self.symbols[sym].position
        if pos.side is None:
            return
        price = self.symbols[sym].price

        # Always use ATR for trail distance — after TP1 the entry-SL spread
        # collapses to ~0 and would freeze the trail.
        atr = self.symbols[sym].atr
        sl_dist = atr * self.cfg.atr_sl_mult if atr > 0 else abs(pos.entry - pos.trail_stop)

        # minimum trail distance to prevent zero-distance freeze
        min_sl_dist = pos.entry * 0.003  # 0.3% minimum
        sl_dist = max(sl_dist, min_sl_dist)

        if sl_dist <= 0:
            return

        # FIX #9: the runner uses its own (tighter) offset consistently.
        # Before, _tp2_exit set a tight runner trail and this method then
        # widened it back on the very next tick.
        if pos.runner_active:
            trail_offset = sl_dist * self.cfg.runner_trail_mult
        else:
            trail_offset = sl_dist * self.cfg.trail_offset_mult
        activation = sl_dist * self.cfg.trail_activation_mult
        profit = (price - pos.entry) if pos.side == "LONG" else (pos.entry - price)

        if not pos.trail_active:
            if profit >= activation:
                pos.trail_active = True
                pos.trail_best = price
                pos.trail_stop = (price - trail_offset if pos.side == "LONG"
                                  else price + trail_offset)
        else:
            if pos.side == "LONG" and price > pos.trail_best:
                pos.trail_best = price
                new_stop = price - trail_offset
                pos.trail_stop = max(pos.trail_stop, new_stop)  # only tighten
            elif pos.side == "SHORT" and price < pos.trail_best:
                pos.trail_best = price
                new_stop = price + trail_offset
                pos.trail_stop = min(pos.trail_stop, new_stop)  # only tighten

    # ── per-tick orchestration ─────────────────────────────────────────────
    async def tick(self, live, state: dict):
        for sym in self.symbol_list:
            self.ensure_symbol_state(sym)
        now = time.time()
        cfg = self.cfg

        if now - state["last_position_sync"] >= cfg.position_sync_interval:
            await self.sync_positions()
            state["last_position_sync"] = now

        if now - self._last_indicator_tick >= cfg.indicator_interval:
            sym = self.symbol_list[self._indicator_index % len(self.symbol_list)]
            await self.update_one_symbol(sym)
            self._indicator_index += 1
            self._last_indicator_tick = now

        # Skip sentiment refresh if symbol in cooldown (wastes API calls)
        stale_sent = min(self.symbol_list,
                         key=lambda s: state["last_sentiment"].get(s, 0.0))
        if (now - state["last_sentiment"].get(stale_sent, 0.0) >= cfg.sentiment_interval
                and stale_sent not in self._ai_inflight
                and time.time() >= self.symbols[stale_sent].cooldown_until):
            self._ai_inflight.add(stale_sent)
            state["last_sentiment"][stale_sent] = now
            self._spawn(self._refresh_sentiment(stale_sent))

        balance = await self.exchange.usdt_balance()
        await self.run_trading_logic(balance)

        # FIX #10: refresh AI analysis for symbols WITH open positions too.
        # compute_signal relies on a fresh AI analysis to fire AI exit
        # signals, but the old gate (`not position.is_open`) meant analysis
        # went stale the moment a trade opened — so AI exits could never
        # trigger. Cooldown still skips symbols with NO position.
        stale_ai = min(self.symbol_list,
                       key=lambda s: self.symbols[s].analysis_ts)
        s_ai = self.symbols[stale_ai]
        if (now - s_ai.analysis_ts >= cfg.analysis_interval
                and ("A:" + stale_ai) not in self._ai_inflight
                and (s_ai.position.is_open or time.time() >= s_ai.cooldown_until)):
            self._ai_inflight.add("A:" + stale_ai)
            s_ai.analysis_ts = now
            self._spawn(self._refresh_analysis(stale_ai))

        if self.telegram.enabled:
            self.telegram.notify_signal_changes(self)
            if now - state.get("last_tg_status", 0.0) >= cfg.tg_status_interval:
                self.telegram.schedule(self.telegram.send_message(
                    self.telegram.status_text(self) + "\n\n" + self.telegram.positions_text(self)))
                state["last_tg_status"] = now
            if now - state.get("last_tg_log", 0.0) >= cfg.tg_log_send_interval:
                self.telegram.schedule(self.telegram.send_document(cfg.log_txt_file, "📄 periodic bot.log"))
                state["last_tg_log"] = now

        if self.headless:
            if now - state.get("last_plain_print", 0.0) >= cfg.headless_status_interval:
                self.dashboard.print_plain_status(balance)
                state["last_plain_print"] = now
        else:
            live.update(self.dashboard.build_layout(balance))

    # ── websocket loop ──────────────────────────────────────────────────────
    async def ws_loop(self, live, state: dict):
        bm = BinanceSocketManager(self.exchange.client)
        while True:
            streams = [f"{s.lower()}@miniTicker" for s in self.symbol_list]
            try:
                async with bm.multiplex_socket(streams) as ms:
                    self.ws_needs_restart = False
                    while True:
                        try:
                            res = await asyncio.wait_for(ms.recv(), timeout=5)
                            data = res.get("data", {})
                            sym = data.get("s", "")
                            if sym in self.symbols and "c" in data:
                                self.symbols[sym].price = float(data["c"])
                                self.symbols[sym].price_ts = time.time()
                        except asyncio.TimeoutError:
                            pass
                        except (asyncio.CancelledError, KeyboardInterrupt):
                            raise

                        await self.tick(live, state)

                        if self.ws_needs_restart:
                            console.print("[magenta]🔁 Re-subscribing WS to new symbols…[/magenta]")
                            break
                        try:
                            await asyncio.sleep(2)
                        except (asyncio.CancelledError, KeyboardInterrupt):
                            raise
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as e:
                console.print(f"[yellow]⚠ WS dropped: {type(e).__name__}: {e}[/yellow]")
                return

    # ── top-level run ─────────────────────────────────────────────────────
    async def run(self):
        self.logger._ensure()
        await self.exchange.connect()

        console.print(Panel.fit(
            f"[bold green]🚀 AI Futures Trader v5.2  |  Leverage: {self.cfg.leverage}x  |  "
            f"Risk: {self.cfg.risk_percent*100:.1f}% / trade  |  "
            f"Active symbols: {len(self.symbol_list)}/{len(self.symbol_pool)}  |  "
            f"Max L:{self.cfg.max_open_longs} S:{self.cfg.max_open_shorts}  |  "
            f"Reshuffle: {self.cfg.symbol_reshuffle_hours}h[/bold green]", style="cyan"))

        if not await self.exchange.verify_access():
            self.telegram.notify("❌ Binance API check failed at startup.", "ERROR")
            await self.exchange.close()
            self.executor.shutdown(wait=False)
            return

        await self.filter_tradable_symbols()
        console.print(f"[dim]🎲 Active symbols: {', '.join(self.symbol_list)}[/dim]")

        console.print("[dim]Pre-loading symbol info…[/dim]")
        await self.exchange.symbol_info(self.symbol_list[0])
        await self.setup_leverage()
        await self.sync_positions()

        console.print("[dim]Loading initial indicators (first symbol)…[/dim]")
        await self.update_one_symbol(self.symbol_list[0])
        self._indicator_index = 1
        self._last_indicator_tick = time.time()

        self.session_start_balance = await self.exchange.usdt_balance()
        console.print(f"[cyan]💰 Session start balance: "
                      f"${self.session_start_balance:,.2f} USDT[/cyan]")

        if self.telegram.enabled:
            console.print("[green]📨 Telegram notifications ENABLED[/green]")
            self.telegram.notify(
                f"🚀 AI Futures Trader v5.2 started\n"
                f"Balance: ${self.session_start_balance:,.2f} USDT | Leverage {self.cfg.leverage}x\n"
                f"Symbols ({len(self.symbol_list)}): {', '.join(self.symbol_list)}\n"
                f"AI-integrated trading active.", "INFO")
        else:
            console.print("[yellow]📭 Telegram disabled (set TELEGRAM_BOT_TOKEN + "
                          "TELEGRAM_CHAT_ID in .env)[/yellow]")

        # stagger initial AI analysis
        for i, sym in enumerate(self.symbol_list):
            self.symbols[sym].analysis_ts = time.time() - (i * (self.cfg.analysis_interval /
                                                                 max(len(self.symbol_list), 1)))

        state = {
            "last_position_sync": 0.0,
            "last_sentiment": {sym: float(i * self.cfg.sentiment_interval /
                                          max(len(self.symbol_list), 1))
                               for i, sym in enumerate(self.symbol_list)},
            "last_tg_status": time.time(),
            "last_tg_log": time.time(),
            "last_plain_print": 0.0,
        }

        poller_task = (asyncio.create_task(self.telegram.command_poller(self))
                       if self.telegram.enabled else None)

        if self.headless:
            console.print("[cyan]🖥  Headless mode (no TTY) — plain status lines. "
                          "Use `tail -f bot.log` too.[/cyan]")

        try:
            live_ctx = _NoopLive() if self.headless else Live(
                console=console, refresh_per_second=2, screen=True)
            with live_ctx as live:
                while True:
                    try:
                        await self.ws_loop(live, state)
                    except (asyncio.CancelledError, KeyboardInterrupt):
                        raise
                    except Exception as e:
                        console.print(f"[red]Loop error: {e}[/red]")

                    # Clean up background tasks before reconnect
                    for t in list(self._bg_tasks):
                        t.cancel()
                    self._bg_tasks.clear()
                    self._ai_inflight.clear()

                    console.print(f"[yellow]🔄 Reconnecting in {self.cfg.reconnect_delay}s…[/yellow]")
                    for _ in range(self.cfg.reconnect_delay):
                        try:
                            live.update(self.dashboard.build_layout(await self.exchange.usdt_balance()))
                        except Exception:
                            pass
                        try:
                            await asyncio.sleep(1)
                        except (asyncio.CancelledError, KeyboardInterrupt):
                            raise
                    await self.exchange.close()
                    try:
                        await self.exchange.connect()
                        console.print("[green]✅ Reconnected.[/green]")
                    except Exception as e:
                        console.print(f"[red]Reconnect failed: {e}[/red]")

        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[bold red]Bot stopped.[/bold red]")
            final = await self.exchange.usdt_balance()
            col = "green" if self.daily_realized_pnl >= 0 else "red"
            console.print(f"[cyan]Session PnL: [{col}]${self.daily_realized_pnl:+.2f}[/] | "
                          f"Final balance: ${final:,.2f} USDT[/cyan]")
            if self.telegram.enabled:
                self.telegram.write_log(f"Bot stopped. Session PnL ${self.daily_realized_pnl:+.2f} | "
                                        f"Final balance ${final:,.2f}", "INFO")
                await self.telegram.send_message(
                    f"🛑 Bot stopped\nSession PnL: ${self.daily_realized_pnl:+.2f}\n"
                    f"Final balance: ${final:,.2f} USDT")
                await self.telegram.send_document(self.cfg.log_txt_file, "📄 final bot.log")
        finally:
            for t in list(self._bg_tasks):
                t.cancel()
            if poller_task:
                poller_task.cancel()
            await self.exchange.close()
            self.executor.shutdown(wait=False)