import asyncio
import time
import hashlib
from collections import deque
from pathlib import Path
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from config import BotConfig

import aiohttp

if TYPE_CHECKING:
    from trading_bot import TradingBot

# Telegram limits
_MSG_HARD_LIMIT = 4096
_CAPTION_LIMIT = 1024
_DOC_TAIL_LIMIT = 8 * 1024 * 1024
_MIN_SEND_INTERVAL = 1.05
_MAX_DEDUP_HISTORY = 100  # Keep last 100 message hashes to prevent spam


class Telegram:
    """Notifications + command poller. Never blocks the trading loop."""

    def __init__(self, cfg: "BotConfig"):
        self.cfg = cfg
        self.enabled = cfg.telegram_enabled
        self._base = f"https://api.telegram.org/bot{cfg.telegram_token}"

        # Shared HTTP session
        self._session: Optional[aiohttp.ClientSession] = None
        self._send_lock: Optional[asyncio.Lock] = None
        self._last_send = 0.0

        # Task tracking with cleanup
        self._tasks: set[asyncio.Task] = set()
        self._task_cleanup_interval = 300  # Clean every 5 min
        self._last_task_cleanup = time.monotonic()

        # Message deduplication
        self._recent_hashes: deque[str] = deque(maxlen=_MAX_DEDUP_HISTORY)
        self._last_signal_tg: dict[str, float] = {}

        # Signal state (moved from TradingBot to reduce coupling)
        self._last_signal_state: dict[str, str] = {}

    # ── lifecycle ──────────────────────────────────────────────────────────
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"Connection": "keep-alive"},
            )
        return self._session

    async def aclose(self):
        for t in list(self._tasks):
            t.cancel()
        self._tasks.clear()
        if self._session and not self._session.closed:
            await self._session.close()

    def _cleanup_tasks(self):
        """Remove completed/cancelled tasks to prevent memory leak."""
        now = time.monotonic()
        if now - self._last_task_cleanup < self._task_cleanup_interval:
            return
        done = {t for t in self._tasks if t.done()}
        self._tasks -= done
        self._last_task_cleanup = now

    # ── async file logging ─────────────────────────────────────────────────
    async def write_log(self, msg: str, level: str = "INFO"):
        """Async file logging - never blocks the event loop."""
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} [{level}] {msg}\n"
        try:
            await asyncio.to_thread(self._write_log_sync, line)
        except Exception:
            pass

    def _write_log_sync(self, line: str):
        try:
            with self.cfg.log_txt_file.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

    # ── deduplication ──────────────────────────────────────────────────────
    def _is_duplicate(self, text: str) -> bool:
        """Check if this message was recently sent (prevents spam)."""
        h = hashlib.md5(text.encode()).hexdigest()[:16]
        if h in self._recent_hashes:
            return True
        self._recent_hashes.append(h)
        return False

    # ── paced POST with retry ──────────────────────────────────────────────
    async def _pace(self):
        if self._send_lock is None:
            self._send_lock = asyncio.Lock()
        async with self._send_lock:
            wait = _MIN_SEND_INTERVAL - (time.monotonic() - self._last_send)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_send = time.monotonic()

    async def _post(self, method: str, *, data=None, retries: int = 1) -> Optional[dict]:
        session = await self._get_session()
        url = f"{self._base}/{method}"
        for attempt in range(retries):
            await self._pace()
            try:
                async with session.post(url, data=data) as r:
                    if r.status == 429:
                        body = await _safe_json(r)
                        retry_after = float(
                            (body or {}).get("parameters", {}).get("retry_after", 1)
                        )
                        await asyncio.sleep(min(retry_after, 30) + 0.5)
                        continue
                    if r.status in (500, 502, 503, 504):
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    if r.status != 200:
                        # Log unexpected status but don't retry forever
                        await self.write_log(
                            f"Telegram API {method} returned {r.status}", "WARN"
                        )
                        return None
                    return await _safe_json(r)
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
                else:
                    await self.write_log(f"Telegram {method} failed: {e}", "ERROR")
        return None

    # ── public send methods with retry and dedup ───────────────────────────
    async def send_message(self, text: str, skip_dedup: bool = False):
        if not self.enabled or not text:
            return
        if not skip_dedup and self._is_duplicate(text):
            return
        result = await self._post("sendMessage", data={
            "chat_id": self.cfg.telegram_chat_id,
            "text": text[:_MSG_HARD_LIMIT],
            "disable_web_page_preview": "true",
            "parse_mode": "HTML",
        })
        return result

    async def send_document(self, path: Path, caption: str = ""):
        if not self.enabled:
            return
        try:
            if not path.exists() or path.stat().st_size == 0:
                return
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception:
            return

        try:
            fh = path.open("rb")
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception:
            return

        try:
            form = aiohttp.FormData()
            form.add_field("chat_id", str(self.cfg.telegram_chat_id))
            if caption:
                form.add_field("caption", caption[:_CAPTION_LIMIT])
            form.add_field(
                "document",
                fh,
                filename=path.name,
                content_type="text/plain",
            )
            await self._post("sendDocument", data=form)
        finally:
            try:
                fh.close()
            except Exception:
                pass

    # ── fire-and-forget with cleanup ───────────────────────────────────────
    def schedule(self, coro):
        if not self.enabled:
            _close_coro(coro)
            return
        self._cleanup_tasks()  # Periodic cleanup
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _close_coro(coro)
            return
        task = loop.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def notify(self, msg: str, level: str = "INFO", skip_dedup: bool = False):
        """Log to file AND push to Telegram (non-blocking)."""
        # Fire both independently
        self.schedule(self.write_log(msg, level))
        self.schedule(self.send_message(msg, skip_dedup=skip_dedup))

    # ── status text builders (enhanced) ────────────────────────────────────
    def status_text(self, bot: "TradingBot") -> str:
        lines = ["<b>📊 Signals & AI Analysis</b>"]
        for s in bot.iter_states():
            tag = ""
            if s.signal == "HOLD" and s.hold_reason:
                tag = f" <i>({s.hold_reason})</i>"
            
            # NEW: Show AI info if available
            ai_info = ""
            if hasattr(s, 'analysis') and s.analysis:
                ai = s.analysis
                analysis_ts = getattr(s, 'analysis_ts', None)
                if analysis_ts is not None:
                    ai_age = time.time() - analysis_ts
                    if ai_age < 300:  # Fresh in last 5 min
                        ai_conf = ai.get('confidence', 0)
                        ai_action = ai.get('action', 'WAIT')
                        ai_bias = ai.get('bias', 'NEUTRAL')
                        ai_info = f" | 🤖 {ai_action}({ai_conf}%) {ai_bias}"
            
            emoji = {"BUY": "🟢", "SHORT": "🔴", "SELL_LONG": "🟠",
                     "COVER_SHORT": "🟠", "HOLD": "⚪"}.get(s.signal, "⚪")
            lines.append(f"{emoji} <b>{s.symbol}</b>: {s.signal}{tag}{ai_info}")
        return "\n".join(lines)

    def positions_text(self, bot: "TradingBot") -> str:
        rows = []
        total_pnl = 0.0
        for s in bot.iter_states():
            p = s.position
            if not p.is_open:
                continue
            
            # FIX: Guard against None values
            pnl_usdt = getattr(p, 'pnl_usdt', 0.0) or 0.0
            pnl_pct = getattr(p, 'pnl_pct', 0.0) or 0.0
            entry = getattr(p, 'entry', 0.0) or 0.0
            sl_price = getattr(p, 'sl_price', 0.0) or 0.0
            tp1_price = getattr(p, 'tp1_price', 0.0) or 0.0
            tp2_price = getattr(p, 'tp2_price', 0.0) or 0.0
            
            total_pnl += pnl_usdt
            
            # NEW: Detailed position info
            trail_status = ""
            if getattr(p, 'trail_active', False):
                trail_stop = getattr(p, 'trail_stop', 0.0) or 0.0
                trail_status = f" 🎯 Trail@{trail_stop:,.4f}"
            
            rr = 0.0
            if sl_price != 0 and entry != 0:
                risk = abs(entry - sl_price)
                reward = abs(tp1_price - entry)
                rr = reward / risk if risk > 0 else 0
            
            rows.append(
                f"• <b>{s.symbol}</b> {p.side}\n"
                f"  Entry: ${entry:,.4f} | PnL: <code>{pnl_pct:+.2f}%</code> (${pnl_usdt:+.2f})\n"
                f"  SL: ${sl_price:,.4f} | TP1: ${tp1_price:,.4f} | TP2: ${tp2_price:,.4f}\n"
                f"  R:R {rr:.1f}{trail_status}"
            )
        
        header = f"<b>📌 Open Positions</b> (Total PnL: <code>${total_pnl:+.2f}</code>)\n\n"
        return header + "\n".join(rows) if rows else "📌 No open positions."

    # NEW: AI analysis text
    def ai_text(self, bot: "TradingBot") -> str:
        lines = ["<b>🤖 AI Analysis</b>"]
        for s in bot.iter_states():
            if not hasattr(s, 'analysis') or not s.analysis:
                continue
            ai = s.analysis
            analysis_ts = getattr(s, 'analysis_ts', None)
            if analysis_ts is None:
                age_str = "unknown"
            else:
                age = time.time() - analysis_ts
                age_str = f"{age:.0f}s ago" if age < 60 else f"{age/60:.0f}m ago"
            
            entry = ai.get('entry', 0)
            sl = ai.get('sl', 0)
            tp1 = ai.get('tp1', 0)
            tp2 = ai.get('tp2', 0)
            
            lines.append(
                f"• <b>{s.symbol}</b> ({age_str})\n"
                f"  Bias: {ai.get('bias', 'N/A')} | Action: {ai.get('action', 'N/A')}\n"
                f"  Confidence: {ai.get('confidence', 0)}%\n"
                f"  Entry: ${entry:,.4f} | SL: ${sl:,.4f}\n"
                f"  TP1: ${tp1:,.4f} | TP2: ${tp2:,.4f}\n"
                f"  Reason: {ai.get('reason', 'N/A')[:80]}"
            )
        return "\n".join(lines) if len(lines) > 1 else "🤖 No AI analysis available."

    # NEW: Performance summary
    def performance_text(self, bot: "TradingBot") -> str:
        total_trades = len(bot.trade_log) if hasattr(bot, 'trade_log') else 0
        if total_trades == 0:
            return "📈 No trades yet."
        
        # FIX: Robust win counting — handle both float and string pnl_usdt
        wins = 0
        for t in bot.trade_log:
            pnl = t.get("pnl_usdt", 0)
            if isinstance(pnl, str):
                pnl = pnl.replace("$", "").replace("+", "").replace(",", "")
                try:
                    pnl = float(pnl)
                except ValueError:
                    pnl = 0.0
            if isinstance(pnl, (int, float)) and pnl > 0:
                wins += 1
        
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        daily_pnl = getattr(bot, 'daily_realized_pnl', 0.0) or 0.0
        balance = 0.0
        if hasattr(bot, 'exchange') and bot.exchange is not None:
            balance = getattr(bot.exchange, 'cached_balance', 0.0) or 0.0
        
        halted = getattr(bot, 'trading_halted', False)
        
        return (
            f"<b>📈 Session Performance</b>\n"
            f"Trades: {total_trades} | Win Rate: {win_rate:.1f}%\n"
            f"Daily PnL: <code>${daily_pnl:+.2f}</code>\n"
            f"Balance: <code>${balance:,.2f}</code> USDT\n"
            f"{'⛔ <b>TRADING HALTED</b>' if halted else '✅ Active'}"
        )

    # ── signal change notifications (fixed coupling) ───────────────────────
    def notify_signal_changes(self, bot: "TradingBot"):
        """Debounced alerts on signal transitions; no duplicate spam."""
        now = time.time()
        for s in bot.iter_states():
            prev = self._last_signal_state.get(s.symbol)
            cur = s.signal
            
            # FIX: Determine the actual signal to display and store
            if cur in ("BUY", "SHORT") and s.hold_reason:
                display_signal = f"{cur} (blocked: {s.hold_reason})"
            else:
                display_signal = cur
            
            if prev is None:
                self._last_signal_state[s.symbol] = display_signal
                continue
            if display_signal == prev:  # FIX: Compare display_signal, not cur
                continue
            if now - self._last_signal_tg.get(s.symbol, 0.0) < self.cfg.tg_signal_debounce:
                continue
                
            self._last_signal_tg[s.symbol] = now
            self._last_signal_state[s.symbol] = display_signal  # FIX: Store display_signal
            
            emoji = {"BUY": "🟢", "SHORT": "🔴", "SELL_LONG": "🟠",
                        "COVER_SHORT": "🟠", "HOLD": "⚪"}.get(cur, "⚪")
            
            # FIX: Show blocked reason in alert
            if cur in ("BUY", "SHORT") and s.hold_reason:
                msg = f"{emoji} <b>{s.symbol}</b>: {prev.split(' (')[0]} → {cur}\n" \
                        f"⚠️ <b>BLOCKED: {s.hold_reason}</b>\n" \
                        f"Price: ${s.price:,.4f}"
            else:
                msg = f"{emoji} <b>{s.symbol}</b>: {prev.split(' (')[0] if ' (' in prev else prev} → {cur}\n" \
                        f"Price: ${s.price:,.4f}"
            
            self.notify(msg, "SIGNAL", skip_dedup=True)  # skip dedup for signals

    # ── command poller (enhanced) ──────────────────────────────────────────
    async def command_poller(self, bot: "TradingBot"):
        if not self.enabled:
            return
        session = await self._get_session()
        offset = None
        backoff = 1.0
        chat_filter = str(self.cfg.telegram_chat_id) if self.cfg.telegram_chat_id else ""

        while True:
            try:
                params = {"timeout": 30, "limit": 10}
                if offset is not None:
                    params["offset"] = offset
                async with session.get(
                    f"{self._base}/getUpdates", params=params,
                    timeout=aiohttp.ClientTimeout(total=40),
                ) as r:
                    if r.status == 401:
                        await self.write_log("Telegram token rejected (401); poller stopping.", "ERROR")
                        return
                    if r.status == 409:
                        await self.write_log("Telegram getUpdates conflict (409); backing off.", "WARN")
                        await asyncio.sleep(min(backoff, 30))
                        backoff = min(backoff * 2, 30)
                        continue
                    if r.status != 200:
                        await asyncio.sleep(min(backoff, 30))
                        backoff = min(backoff * 2, 30)
                        continue
                        
                    data = await _safe_json(r)

                if not data or not data.get("ok"):
                    await asyncio.sleep(min(backoff, 30))
                    backoff = min(backoff * 2, 30)
                    continue
                backoff = 1.0

                for upd in data.get("result", []):
                    offset = upd["update_id"] + 1
                    msg = upd.get("message") or {}
                    chat = str(msg.get("chat", {}).get("id", ""))
                    if chat_filter and chat != chat_filter:
                        continue
                    
                    cmd = _parse_command(msg.get("text") or "")
                    
                    # NEW: Emergency commands
                    if cmd == "/halt":
                        bot.trading_halted = True
                        await self.send_message("⛔ <b>Trading HALTED</b>\nNo new positions will be opened.")
                        await self.write_log("Trading halted via Telegram command", "WARN")
                        
                    elif cmd == "/resume":
                        bot.trading_halted = False
                        bot.daily_realized_pnl = 0.0  # Reset daily loss tracker
                        bot.session_start_balance = getattr(bot.exchange, 'cached_balance', 0.0) if hasattr(bot, 'exchange') else 0.0
                        await self.send_message("✅ <b>Trading RESUMED</b>\nDaily loss counter reset.")
                        await self.write_log("Trading resumed via Telegram command", "INFO")
                        
                    elif cmd == "/closeall":
                        await self.send_message("🔴 <b>Closing ALL positions...</b>")
                        for s in bot.iter_states():
                            if s.position.is_open:
                                # FIX: Track the close tasks
                                task = asyncio.create_task(bot.close_position(
                                    s.symbol, reason="EMERGENCY_CLOSE"
                                ))
                                self._tasks.add(task)
                                task.add_done_callback(self._tasks.discard)
                        await self.write_log("Emergency close all triggered via Telegram", "WARN")
                        
                    elif cmd in ("/log", "/logs"):
                        await self.send_document(self.cfg.log_txt_file, "📄 bot.log")
                        
                    elif cmd == "/status":
                        await self.send_message(self.status_text(bot))
                        
                    elif cmd in ("/positions", "/pos"):
                        await self.send_message(self.positions_text(bot))
                        
                    elif cmd in ("/balance", "/bal"):
                        balance = 0.0
                        if hasattr(bot, 'exchange') and bot.exchange is not None:
                            balance = getattr(bot.exchange, 'cached_balance', 0.0) or 0.0
                        daily_pnl = getattr(bot, 'daily_realized_pnl', 0.0) or 0.0
                        halted = getattr(bot, 'trading_halted', False)
                        await self.send_message(
                            f"💰 <b>Balance</b>: <code>${balance:,.2f}</code> USDT\n"
                            f"📈 Daily PnL: <code>${daily_pnl:+.2f}</code>\n"
                            f"{'⛔ HALTED' if halted else '✅ Active'}"
                        )
                        
                    elif cmd == "/ai":
                        await self.send_message(self.ai_text(bot))
                        
                    elif cmd == "/perf":
                        await self.send_message(self.performance_text(bot))
                        
                    elif cmd in ("/start", "/help"):
                        await self.send_message(
                            "🤖 <b>AI Futures Trader Commands</b>\n\n"
                            "<b>Monitoring:</b>\n"
                            "/status – signals & AI analysis\n"
                            "/positions – open trades with SL/TP\n"
                            "/balance – balance & daily PnL\n"
                            "/ai – current AI consensus\n"
                            "/perf – session performance\n\n"
                            "<b>Control:</b>\n"
                            "/halt – stop new entries\n"
                            "/resume – resume trading\n"
                            "/closeall – emergency close all\n\n"
                            "<b>Logs:</b>\n"
                            "/log – send bot.log file"
                        )
                        
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as e:
                await self.write_log(f"Command poller error: {e}", "ERROR")
                await asyncio.sleep(min(backoff, 30))
                backoff = min(backoff * 2, 30)


# ── module helpers ──────────────────────────────────────────────────────────
def _parse_command(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    token = text.split(maxsplit=1)[0]
    token = token.split("@", 1)[0]
    return token.lower()


def _read_tail_streaming(path: Path, max_bytes: int) -> bytes:
    """Memory-efficient: seek to end and read only last max_bytes."""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                # Skip to next newline to avoid partial line
                f.readline()
                # FIX: Use actual newline, not escaped backslash-n
                header = f"...[truncated to last {max_bytes // 1024} KB]...\n".encode()
                return header + f.read()
            return f.read()
    except Exception:
        return b""


async def _safe_json(resp) -> Optional[dict]:
    try:
        return await resp.json(content_type=None)
    except Exception:
        return None


def _close_coro(coro):
    try:
        coro.close()
    except Exception:
        pass