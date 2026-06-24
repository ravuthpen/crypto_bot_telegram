"""
AI Futures Trader — Production Grade v4
=========================================
v4 changes:
  • Random symbol rotation — SYMBOLS shuffled at startup + re-shuffled every
    SYMBOL_RESHUFFLE_HOURS so every coin gets equal indicator/analysis attention
  • Flexible open positions — MAX_OPEN_LONGS and MAX_OPEN_SHORTS configured
    independently; bot can hold longs and shorts simultaneously across symbols
  • Signal-flip (reverse trade) — if a BUY fires on a symbol that already holds
    a SHORT (or vice-versa), the existing position is closed first then the new
    one opened immediately in the same tick
  • Per-side position counter — count_open_longs() / count_open_shorts() replace
    the single count_open_positions() so each side has its own slot budget
  • All v3 smooth-loading fixes retained
"""

import os
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

import asyncio
import csv
import math
import random
import re
import time
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import indicators as ta   # self-contained indicators (replaces pandas_ta)
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.columns import Columns
from rich.console import Group as RichGroup
from binance import AsyncClient, BinanceSocketManager
from binance.enums import (
    SIDE_BUY, SIDE_SELL,
    ORDER_TYPE_MARKET,
    FUTURE_ORDER_TYPE_STOP_MARKET,
)

load_dotenv()

# ╔══════════════════════════════════════════════════════════════════╗
# ║                           CONFIG                               ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── Master symbol pool — bot picks randomly from here ────────────────────────
# NOTE: at startup the bot filters this against Binance's live tradable symbols
# (filter_tradable_symbols), so a delisted entry here is dropped automatically.
_SYMBOL_POOL = [
    "AVAXUSDT", "BTCUSDT",  "ETHUSDT",  "BNBUSDT",     "SOLUSDT",
    "XRPUSDT",  "DOGEUSDT", "ADAUSDT",  "LINKUSDT",    "LTCUSDT",
    "TRXUSDT",  "DOTUSDT",  "POLUSDT",  "ATOMUSDT",    "NEARUSDT",
    "ARBUSDT",  "OPUSDT",   "APTUSDT",  "SUIUSDT",     "1000PEPEUSDT",
    "BNXUSDT",  "INJUSDT",  "TIAUSDT",  "SEIUSDT",     "WIFUSDT",
    "JUPUSDT",  "PYTHUSDT", "STRKUSDT", "FETUSDT",     "RENDERUSDT",
]
# How many symbols to actively watch at once (randomly selected from pool)
ACTIVE_SYMBOL_COUNT    = 20
# Reshuffle the active symbol list every N hours
SYMBOL_RESHUFFLE_HOURS = 4.0

# Initialise active list (shuffled)
SYMBOLS: list[str] = random.sample(_SYMBOL_POOL, min(ACTIVE_SYMBOL_COUNT, len(_SYMBOL_POOL)))
MAIN_SYMBOL = SYMBOLS[0]

USE_TESTNET             = False
LEVERAGE                = 3
RISK_PERCENT            = 0.01
ATR_SL_MULT             = 1.5
ATR_TP1_MULT            = 2.0
ATR_TP2_MULT            = 3.5
TRAIL_ACTIVATION_MULT   = 1.2
TRAIL_OFFSET_MULT       = 0.8

# ── Per-side position limits (independent) ───────────────────────────────────
MAX_OPEN_LONGS          = 3   # max simultaneous LONG positions across all symbols
MAX_OPEN_SHORTS         = 3   # max simultaneous SHORT positions across all symbols

MIN_USDT_TO_TRADE       = 20.0
ADX_MIN                 = 20.0
SQUEEZE_LOOKBACK        = 20
BB_STD                  = 2.0
PARTIAL_CLOSE_FRACTION  = 0.50
COOLDOWN_AFTER_LOSS     = 300
DAILY_LOSS_LIMIT_PCT    = 0.05
STALE_PRICE_THRESHOLD   = 60
ANALYSIS_INTERVAL       = 90
POSITION_SYNC_INTERVAL  = 8
BALANCE_REFRESH_INTERVAL= 30
INDICATOR_INTERVAL      = 30
SENTIMENT_INTERVAL      = 300
RECONNECT_DELAY         = 5
TF_PRIMARY              = "15m"
TF_CONFIRM              = "1h"
LOG_FILE                = Path("trades.csv")

# ── Display mode ─────────────────────────────────────────────────────────────
# The rich Live dashboard needs a real interactive terminal. In Docker / piped
# output it just produces blank lines, so we auto-detect and fall back to plain
# periodic log lines. Force either mode with the DASHBOARD env var: "1" / "0".
import sys as _sys
_dash_env = os.getenv("DASHBOARD", "")
if _dash_env in ("0", "false", "False"):
    HEADLESS = True
elif _dash_env in ("1", "true", "True"):
    HEADLESS = False
else:
    HEADLESS = not _sys.stdout.isatty()   # auto: no TTY (Docker) → headless
HEADLESS_STATUS_INTERVAL = 30   # seconds between plain status prints when headless

# ── Telegram notifications ───────────────────────────────────────────────────
# Set these in your .env file:
#   TELEGRAM_BOT_TOKEN=123456:ABC...      (from @BotFather)
#   TELEGRAM_CHAT_ID=123456789            (your user/group id; get via @userinfobot)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ENABLED   = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
LOG_TXT_FILE       = Path("bot.log")    # human-readable event log (pushed to Telegram)
STATUS_INTERVAL    = 1800   # seconds between Telegram status summaries (signals + positions)
LOG_SEND_INTERVAL  = 3600   # seconds between automatic bot.log document pushes
TG_SIGNAL_DEBOUNCE = 60     # min seconds between signal-change alerts per symbol
POSITION_ADOPT_GRACE = 10   # sec: don't clear/adopt within this of a local open (open→sync race)

# ╔══════════════════════════════════════════════════════════════════╗
# ║                          CLIENTS                               ║
# ╚══════════════════════════════════════════════════════════════════╝

AI_client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url=os.getenv("URL", "https://api.x.ai/v1"),
)
console   = Console()
_executor = ThreadPoolExecutor(max_workers=4)   # for blocking AI calls

# ╔══════════════════════════════════════════════════════════════════╗
# ║                        GLOBAL STATE                            ║
# ╚══════════════════════════════════════════════════════════════════╝

def _empty_position() -> dict:
    return {
        "side":         None,
        "qty":          0.0,
        "entry":        0.0,
        "sl_price":     0.0,
        "tp1_price":    0.0,
        "tp2_price":    0.0,
        "trail_active": False,
        "trail_best":   0.0,
        "trail_stop":   0.0,
        "pnl_pct":      0.0,
        "pnl_usdt":     0.0,
        "tp1_hit":      False,
        "sl_order_id":  None,
        "tp1_order_id": None,
        "open_time":    0.0,
    }

market_data: dict[str, dict] = {sym: {
    "price":      0.0,
    "price_ts":   0.0,
    "rsi_15":     50.0,
    "rsi_1h":     50.0,
    "sma20":      0.0,
    "ema50":      0.0,
    "ema200":     0.0,
    "macd":       0.0,
    "macd_sig":   0.0,
    "atr":        0.0,
    "adx":        0.0,
    "bb_squeeze": False,
    "volume_ok":  False,
    "signal":     "HOLD",
    "hold_reason":"",      # which gate blocked an entry (shown next to HOLD)
    "sentiment":  "NEUTRAL",
    "ind_ts":     0.0,      # timestamp of last successful indicator update
    "position":   _empty_position(),
} for sym in SYMBOLS}

analyses:       dict[str, str]   = {sym: "🔄 Initialising…" for sym in SYMBOLS}
analysis_time:  dict[str, float] = {sym: 0.0 for sym in SYMBOLS}
cooldown_until: dict[str, float] = {sym: 0.0 for sym in SYMBOLS}
trade_log:      list[dict]       = []

# ── Cached balance (avoid fetching every 2 s) ────────────────────────────────
_cached_balance:          float = 0.0
_cached_balance_ts:       float = 0.0

# ── Symbol-info cache (avoid repeated exchange_info calls) ───────────────────
_sym_info_cache: dict[str, dict] = {}

# ── Staggered indicator rotation ─────────────────────────────────────────────
_indicator_index: int    = 0
_last_indicator_tick: float = 0.0

# ── Symbol reshuffle timer ────────────────────────────────────────────────────
_last_reshuffle: float = time.time()
_ws_needs_restart: bool = False   # set True by reshuffle → WS re-subscribes

# ── Leverage tracking (so reshuffled symbols also get correct leverage) ──────
_leverage_done: set[str] = set()

# ── Daily P&L ────────────────────────────────────────────────────────────────
_session_start_balance: float = 0.0
_daily_realized_pnl:    float = 0.0
_trading_halted:        bool  = False

# ── Telegram signal-change tracking ──────────────────────────────────────────
_last_signal:    dict[str, str]   = {}   # last seen signal per symbol
_last_signal_tg: dict[str, float] = {}   # last TG alert time per symbol (debounce)

# ╔══════════════════════════════════════════════════════════════════╗
# ║                      CSV LOGGER                                ║
# ╚══════════════════════════════════════════════════════════════════╝

_LOG_FIELDS = [
    "date","time","symbol","action","side","entry","exit",
    "qty","sl","tp1","tp2","pnl_pct","pnl_usdt","reason",
]

def _ensure_log():
    if not LOG_FILE.exists():
        with LOG_FILE.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=_LOG_FIELDS).writeheader()

def log_trade(row: dict):
    _ensure_log()
    safe = {k: row.get(k, "") for k in _LOG_FIELDS}
    with LOG_FILE.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=_LOG_FIELDS).writerow(safe)

# ╔══════════════════════════════════════════════════════════════════╗
# ║                     TELEGRAM                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

def write_log(msg: str, level: str = "INFO"):
    """Append a timestamped line to bot.log (the file pushed to Telegram)."""
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} [{level}] {msg}"
    try:
        with LOG_TXT_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


async def tg_send_message(text: str):
    """Send a plain-text message to Telegram (no parse_mode → safe for '<' in reasons)."""
    if not TELEGRAM_ENABLED:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text[:4000],                  # Telegram hard limit is 4096
        "disable_web_page_preview": "true",
    }
    try:
        async with aiohttp.ClientSession() as s:
            await s.post(url, data=payload, timeout=aiohttp.ClientTimeout(total=10))
    except Exception:
        pass   # never let a Telegram failure crash the bot


async def tg_send_document(path: Path, caption: str = ""):
    """Upload a file (e.g. bot.log) to Telegram as a document."""
    if not TELEGRAM_ENABLED or not path.exists():
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
    try:
        content = path.read_bytes()
        if not content:
            return
        form = aiohttp.FormData()
        form.add_field("chat_id", str(TELEGRAM_CHAT_ID))
        if caption:
            form.add_field("caption", caption[:1000])
        form.add_field("document", content, filename=path.name,
                       content_type="text/plain")
        async with aiohttp.ClientSession() as s:
            await s.post(url, data=form, timeout=aiohttp.ClientTimeout(total=30))
    except Exception:
        pass


def tg_schedule(coro):
    """Fire-and-forget a Telegram coroutine so the trading loop never blocks on it."""
    if not TELEGRAM_ENABLED:
        # close the coroutine to avoid 'never awaited' warnings
        try: coro.close()
        except Exception: pass
        return
    try:
        asyncio.get_event_loop().create_task(coro)
    except RuntimeError:
        try: coro.close()
        except Exception: pass


def notify(msg: str, level: str = "INFO"):
    """Log an event to bot.log AND push it to Telegram (non-blocking)."""
    write_log(msg, level)
    tg_schedule(tg_send_message(msg))


def _tg_status_text() -> str:
    lines = ["📊 Signals"]
    for sym in SYMBOLS:
        d   = market_data[sym]
        sig = d["signal"]
        r   = d.get("hold_reason", "")
        tag = f" ({r})" if sig == "HOLD" and r else ""
        lines.append(f"• {sym}: {sig}{tag}")
    return "\n".join(lines)


def _tg_positions_text() -> str:
    rows = []
    for sym in SYMBOLS:
        pos = market_data[sym]["position"]
        if pos["side"]:
            rows.append(
                f"• {sym} {pos['side']} @ ${pos['entry']:,.4f} | "
                f"PnL {pos['pnl_pct']:+.2f}% (${pos['pnl_usdt']:+.2f})"
            )
    if not rows:
        return "📌 No open positions."
    return "📌 Open positions\n" + "\n".join(rows)


def notify_signal_changes():
    """
    Detect signal transitions (HOLD→BUY, BUY→HOLD, …) and push a debounced
    Telegram alert for each. Avoids the per-tick HOLD spam of 20 symbols.
    """
    now = time.time()
    for sym in SYMBOLS:
        cur  = market_data[sym]["signal"]
        prev = _last_signal.get(sym)
        if cur == prev:
            continue
        _last_signal[sym] = cur
        if prev is None:
            continue   # skip baseline on first sight (avoids startup burst)
        if now - _last_signal_tg.get(sym, 0.0) < TG_SIGNAL_DEBOUNCE:
            continue
        _last_signal_tg[sym] = now
        d     = market_data[sym]
        r     = d.get("hold_reason", "")
        tag   = f" ({r})" if cur == "HOLD" and r else ""
        emoji = {"BUY": "🟢", "SHORT": "🔴", "SELL_LONG": "🟠",
                 "COVER_SHORT": "🟠", "HOLD": "⚪"}.get(cur, "⚪")
        msg = f"{emoji} {sym}: {prev} → {cur}{tag}  @ ${d['price']:,.4f}"
        write_log(msg, "SIGNAL")
        tg_schedule(tg_send_message(msg))


async def telegram_command_poller():
    """
    Long-poll Telegram for commands. Read-only — never touches the binance
    client (uses cached state) so it's safe to run beside the trading loop.
    Commands: /status /positions /balance /log /help
    """
    if not TELEGRAM_ENABLED:
        return
    base   = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    offset = None
    async with aiohttp.ClientSession() as s:
        while True:
            try:
                params = {"timeout": 30}
                if offset is not None:
                    params["offset"] = offset
                async with s.get(f"{base}/getUpdates", params=params,
                                 timeout=aiohttp.ClientTimeout(total=40)) as r:
                    data = await r.json()

                for upd in data.get("result", []):
                    offset = upd["update_id"] + 1
                    msg    = upd.get("message") or {}
                    chat   = str(msg.get("chat", {}).get("id", ""))
                    text   = (msg.get("text") or "").strip().lower()
                    # only respond to the configured chat
                    if str(TELEGRAM_CHAT_ID) and chat != str(TELEGRAM_CHAT_ID):
                        continue
                    if text in ("/log", "/logs"):
                        await tg_send_document(LOG_TXT_FILE, "📄 bot.log")
                    elif text == "/status":
                        await tg_send_message(_tg_status_text())
                    elif text in ("/positions", "/pos"):
                        await tg_send_message(_tg_positions_text())
                    elif text in ("/balance", "/bal"):
                        await tg_send_message(
                            f"💰 Balance (cached): ${_cached_balance:,.2f} USDT\n"
                            f"📈 Daily PnL: ${_daily_realized_pnl:+.2f}\n"
                            f"{'⛔ HALTED' if _trading_halted else '✅ Active'}"
                        )
                    elif text in ("/start", "/help"):
                        await tg_send_message(
                            "🤖 AI Futures Trader commands:\n"
                            "/status – current signals (incl. HOLD reasons)\n"
                            "/positions – open trades & PnL\n"
                            "/balance – balance & daily PnL\n"
                            "/log – send bot.log file"
                        )
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception:
                await asyncio.sleep(5)   # transient error → back off and retry

# ╔══════════════════════════════════════════════════════════════════╗
# ║                    EXCHANGE HELPERS                            ║
# ╚══════════════════════════════════════════════════════════════════╝

async def setup_leverage(client: AsyncClient):
    for sym in SYMBOLS:
        await ensure_leverage(client, sym)


async def ensure_leverage(client: AsyncClient, symbol: str):
    """Set leverage for a symbol once; safe to call repeatedly (cached)."""
    if symbol in _leverage_done:
        return
    try:
        await client.futures_change_leverage(symbol=symbol, leverage=LEVERAGE)
    except Exception as e:
        console.print(f"[yellow]⚠ leverage {symbol}: {e}[/yellow]")
    _leverage_done.add(symbol)   # mark done even on failure to avoid retry spam


async def verify_binance_access(client: AsyncClient) -> bool:
    """
    Confirm the API key works and has Futures permission BEFORE the bot starts
    trading. Returns True on success; on failure prints a clear, actionable
    message and returns False so main() can exit instead of spamming errors.
    """
    console.print("[dim]Verifying Binance API access…[/dim]")
    try:
        # futures_account_balance requires a valid key WITH Futures permission
        bals = await client.futures_account_balance()
        usdt = next((float(b["balance"]) for b in bals if b["asset"] == "USDT"), 0.0)
        console.print(f"[green]✅ Binance API OK — Futures USDT balance: ${usdt:,.2f}[/green]")
        return True
    except Exception as e:
        msg = str(e)
        console.print(Panel.fit(
            f"[bold red]❌ Binance API check failed[/bold red]\n\n"
            f"{msg}\n\n"
            f"[yellow]If this is code -2015 'Invalid API-key, IP, or permissions':[/yellow]\n"
            f"  1. Key/secret correct and matches USE_TESTNET={USE_TESTNET}?\n"
            f"     (testnet keys only work with testnet=True, and vice-versa)\n"
            f"  2. Is [bold]Futures[/bold] permission enabled on the API key?\n"
            f"  3. Is this server's IP in the key's allowed-IP list?\n"
            f"  4. On a VPS, your egress IP may differ — check it with:\n"
            f"     curl -s https://api.ipify.org\n",
            style="red",
        ))
        notify(f"❌ Binance API check failed at startup: {msg[:200]}", "ERROR")
        return False


async def get_symbol_info(client: AsyncClient, symbol: str) -> dict:
    """Return LOT_SIZE / PRICE_FILTER for a symbol. Results are cached forever."""
    if symbol in _sym_info_cache:
        return _sym_info_cache[symbol]
    try:
        info = await client.futures_exchange_info()
        for s in info["symbols"]:
            sym = s["symbol"]
            f   = {x["filterType"]: x for x in s["filters"]}
            _sym_info_cache[sym] = {
                "step_size":    float(f["LOT_SIZE"]["stepSize"]),
                "tick_size":    float(f["PRICE_FILTER"]["tickSize"]),
                "min_qty":      float(f["LOT_SIZE"]["minQty"]),
                "min_notional": float(f.get("MIN_NOTIONAL", {}).get("notional", 5)),
            }
        if symbol in _sym_info_cache:
            return _sym_info_cache[symbol]
    except Exception as e:
        console.print(f"[yellow]⚠ get_symbol_info: {e}[/yellow]")
    return {"step_size": 0.001, "tick_size": 0.01, "min_qty": 0.001, "min_notional": 5.0}


async def filter_tradable_symbols(client: AsyncClient):
    """
    Query Binance for the set of symbols that are CURRENTLY trading
    (status == 'TRADING', PERPETUAL, USDT-margined) and prune the pool +
    active list of anything delisted (e.g. MATICUSDT → POLUSDT). This makes the
    bot self-healing: a dead symbol can never sit at $0.0000 forever.
    """
    global SYMBOLS, MAIN_SYMBOL, _SYMBOL_POOL
    try:
        info = await client.futures_exchange_info()
    except Exception as e:
        console.print(f"[yellow]⚠ Could not verify symbols (keeping pool as-is): {e}[/yellow]")
        return

    tradable = {
        s["symbol"]
        for s in info.get("symbols", [])
        if s.get("status") == "TRADING"
        and s.get("contractType") == "PERPETUAL"
        and s.get("quoteAsset") == "USDT"
    }

    pool_before = list(_SYMBOL_POOL)
    dead = [s for s in pool_before if s not in tradable]
    live_pool = [s for s in pool_before if s in tradable]

    if dead:
        console.print(f"[yellow]⚠ Dropping {len(dead)} delisted/untradable symbol(s): "
                      f"{', '.join(dead)}[/yellow]")
    if not live_pool:
        console.print("[red]❌ No tradable symbols left in pool! Keeping original list.[/red]")
        return

    _SYMBOL_POOL[:] = live_pool

    # Re-pick the active SYMBOLS from the cleaned pool (keep any still valid)
    kept = [s for s in SYMBOLS if s in tradable]
    need = max(0, min(ACTIVE_SYMBOL_COUNT, len(live_pool)) - len(kept))
    extra_choices = [s for s in live_pool if s not in kept]
    random.shuffle(extra_choices)
    SYMBOLS[:] = kept + extra_choices[:need]
    if SYMBOLS:
        MAIN_SYMBOL = SYMBOLS[0]

    console.print(f"[green]✅ {len(live_pool)} tradable symbols verified; "
                  f"{len(SYMBOLS)} active.[/green]")


def _round_step(value: float, step: float) -> float:
    """Round value down to the symbol's step/tick size (LOT_SIZE / PRICE_FILTER)."""
    if step == 0:
        return value
    precision = max(0, round(-math.log10(step)))
    return round(round(value / step) * step, precision)

async def _adopt_position(
    client: AsyncClient, symbol: str, side: str,
    qty: float, entry: float, unrealized: float, pnl_pct: float,
):
    """
    Reconstruct full management state for a position found on the exchange that
    the bot isn't tracking (manual open, restart, or side flip), so the software
    SL/TP/trail engine can manage it automatically. Also ensures a hard
    catastrophic SL exists on the exchange for the adopted size.
    """
    d   = market_data[symbol]
    atr = d["atr"] if d["atr"] > 0 else entry * 0.005   # fallback if indicators not loaded yet

    sl_distance  = atr * ATR_SL_MULT
    tp1_distance = atr * ATR_TP1_MULT
    tp2_distance = atr * ATR_TP2_MULT

    if side == "LONG":
        sl_price, tp1_price, tp2_price = (
            entry - sl_distance, entry + tp1_distance, entry + tp2_distance)
    else:
        sl_price, tp1_price, tp2_price = (
            entry + sl_distance, entry - tp1_distance, entry - tp2_distance)

    info      = await get_symbol_info(client, symbol)
    sl_price  = _round_step(sl_price,  info["tick_size"])
    tp1_price = _round_step(tp1_price, info["tick_size"])
    tp2_price = _round_step(tp2_price, info["tick_size"])

    # Replace any stray orders with a fresh hard SL for the actual adopted size
    await cancel_open_orders(client, symbol)
    sl_id, _ = await place_bracket_orders(client, symbol, side, qty, sl_price, tp1_price)

    market_data[symbol]["position"] = {
        **_empty_position(),
        "side":         side,
        "qty":          qty,
        "entry":        entry,
        "sl_price":     sl_price,
        "tp1_price":    tp1_price,
        "tp2_price":    tp2_price,
        "trail_best":   entry,
        "trail_stop":   sl_price,
        "pnl_pct":      pnl_pct,
        "pnl_usdt":     unrealized,
        "sl_order_id":  sl_id,
        "open_time":    time.time(),
    }
    console.print(
        f"[cyan]🧬 Adopted {side} {symbol} @ ${entry:,.4f} | "
        f"SL ${sl_price:,.4f}  TP1 ${tp1_price:,.4f}  TP2 ${tp2_price:,.4f}[/cyan]"
    )
    notify(
        f"🧬 Adopted existing {side} {symbol} @ ${entry:,.4f}\n"
        f"SL ${sl_price:,.4f} | TP1 ${tp1_price:,.4f} | TP2 ${tp2_price:,.4f}",
        "TRADE",
    )

async def sync_positions(client: AsyncClient):
    try:
        positions = await client.futures_position_information()
        for p in positions:
            sym = p["symbol"]
            if sym not in market_data:
                continue

            qty = float(p.get("positionAmt", 0))
            pos = market_data[sym]["position"]
            now = time.time()

            # ── Exchange flat → clear local state ─────────────────────────────
            if qty == 0:
                # Grace guard: a just-opened position may not show up yet.
                if (pos["side"] is not None
                        and now - pos.get("open_time", 0.0) >= POSITION_ADOPT_GRACE):
                    market_data[sym]["position"] = _empty_position()
                continue

            entry      = float(p.get("entryPrice", 0))
            unrealized = float(p.get("unRealizedProfit", 0))
            pnl_pct    = (unrealized / (abs(qty) * entry) * 100) if entry > 0 else 0.0
            side       = "LONG" if qty > 0 else "SHORT"

            # Untracked = manual open, post-restart recovery, or a side flip the
            # bot didn't register. Anything with no SL level can't be managed.
            untracked = (
                pos["side"] is None
                or pos["side"] != side
                or pos["sl_price"] <= 0
            )

            if untracked:
                # Don't fight a position we opened microseconds ago.
                if now - pos.get("open_time", 0.0) < POSITION_ADOPT_GRACE:
                    pos.update({"qty": abs(qty), "entry": entry,
                                "pnl_pct": pnl_pct, "pnl_usdt": unrealized})
                    continue
                await _adopt_position(client, sym, side, abs(qty),
                                      entry, unrealized, pnl_pct)
            else:
                # Already managed → just refresh live numbers.
                pos.update({
                    "qty":      abs(qty),
                    "entry":    entry,
                    "pnl_pct":  pnl_pct,
                    "pnl_usdt": unrealized,
                })
    except Exception as e:
        console.print(f"[yellow]⚠ sync_positions: {e}[/yellow]")


async def get_usdt_balance(client: AsyncClient) -> float:
    """Return cached balance; refresh every BALANCE_REFRESH_INTERVAL seconds."""
    global _cached_balance, _cached_balance_ts
    if time.time() - _cached_balance_ts < BALANCE_REFRESH_INTERVAL:
        return _cached_balance
    try:
        for b in await client.futures_account_balance():
            if b["asset"] == "USDT":
                _cached_balance    = float(b["balance"])
                _cached_balance_ts = time.time()
                return _cached_balance
    except Exception as e:
        console.print(f"[yellow]⚠ balance: {e}[/yellow]")
    return _cached_balance


async def cancel_open_orders(client: AsyncClient, symbol: str):
    try:
        await client.futures_cancel_all_open_orders(symbol=symbol)
    except Exception:
        pass


async def _place_market(client: AsyncClient, symbol: str, side: str, qty: float) -> bool:
    try:
        await client.futures_create_order(
            symbol=symbol, side=side.upper(),
            type=ORDER_TYPE_MARKET, quantity=qty,
        )
        col = "green" if side.upper() == "BUY" else "red"
        console.print(f"[bold {col}]✅ {side.upper()} {symbol} qty={qty}[/]")
        return True
    except Exception as e:
        console.print(f"[red]❌ order failed {symbol} {side}: {e}[/red]")
        return False


async def place_bracket_orders(
    client: AsyncClient, symbol: str, pos_side: str,
    qty: float, sl_price: float, tp1_price: float,
) -> tuple[int | None, int | None]:
    """
    Place ONLY the hard stop-loss on the exchange (catastrophic protection that
    survives a bot crash). TP1 / TP2 / trailing are managed in software to avoid
    a double-close race between an exchange TP order and the software TP check.
    Returns (sl_order_id, None).
    """
    close_side = "SELL" if pos_side == "LONG" else "BUY"
    info  = await get_symbol_info(client, symbol)
    sl_p  = _round_step(sl_price, info["tick_size"])
    sl_id = None

    try:
        r = await client.futures_create_order(
            symbol=symbol, side=close_side,
            type=FUTURE_ORDER_TYPE_STOP_MARKET,
            stopPrice=sl_p, closePosition=True,
            timeInForce="GTE_GTC", workingType="MARK_PRICE",
        )
        sl_id = r.get("orderId")
    except Exception as e:
        console.print(f"[yellow]⚠ SL order {symbol}: {e}[/yellow]")

    return sl_id, None


async def replace_exchange_sl(client: AsyncClient, symbol: str, pos_side: str, new_sl: float) -> int | None:
    """Cancel existing SL/TP orders and place a fresh stop-loss (used to move SL to breakeven)."""
    close_side = "SELL" if pos_side == "LONG" else "BUY"
    info = await get_symbol_info(client, symbol)
    sl_p = _round_step(new_sl, info["tick_size"])
    await cancel_open_orders(client, symbol)
    try:
        r = await client.futures_create_order(
            symbol=symbol, side=close_side,
            type=FUTURE_ORDER_TYPE_STOP_MARKET,
            stopPrice=sl_p, closePosition=True,
            timeInForce="GTE_GTC", workingType="MARK_PRICE",
        )
        return r.get("orderId")
    except Exception as e:
        console.print(f"[yellow]⚠ SL replace {symbol}: {e}[/yellow]")
        return None


async def open_position(client: AsyncClient, symbol: str, side: str, balance: float) -> bool:
    await ensure_leverage(client, symbol)   # guarantees correct leverage incl. reshuffled symbols

    d     = market_data[symbol]
    price = d["price"]
    atr   = d["atr"] if d["atr"] > 0 else price * 0.005

    sl_distance  = atr * ATR_SL_MULT
    tp1_distance = atr * ATR_TP1_MULT
    tp2_distance = atr * ATR_TP2_MULT

    risk_usdt = balance * RISK_PERCENT
    raw_qty   = (risk_usdt * LEVERAGE) / sl_distance
    info      = await get_symbol_info(client, symbol)
    qty       = _round_step(raw_qty, info["step_size"])
    qty       = max(qty, info["min_qty"])

    if qty * price < info["min_notional"]:
        console.print(f"[yellow]⚠ {symbol} notional too small ({qty*price:.2f} USDT)[/yellow]")
        return False

    order_side = SIDE_BUY if side == "LONG" else SIDE_SELL
    if not await _place_market(client, symbol, order_side, qty):
        return False

    sl_price  = price - sl_distance  if side == "LONG" else price + sl_distance
    tp1_price = price + tp1_distance if side == "LONG" else price - tp1_distance
    tp2_price = price + tp2_distance if side == "LONG" else price - tp2_distance

    sl_price  = _round_step(sl_price,  info["tick_size"])
    tp1_price = _round_step(tp1_price, info["tick_size"])
    tp2_price = _round_step(tp2_price, info["tick_size"])

    sl_id, tp1_id = await place_bracket_orders(
        client, symbol, side, qty, sl_price, tp1_price
    )

    now = time.time()
    d["position"].update({
        "side":         side,
        "qty":          qty,
        "entry":        price,
        "sl_price":     sl_price,
        "tp1_price":    tp1_price,
        "tp2_price":    tp2_price,
        "trail_active": False,
        "trail_best":   price,
        "trail_stop":   sl_price,
        "pnl_pct":      0.0,
        "pnl_usdt":     0.0,
        "tp1_hit":      False,
        "sl_order_id":  sl_id,
        "tp1_order_id": tp1_id,
        "open_time":    now,
    })
    # Invalidate cached balance so next read is fresh
    global _cached_balance_ts
    _cached_balance_ts = 0.0

    row = {
        "date": date.today().isoformat(), "time": datetime.now().strftime("%H:%M:%S"),
        "symbol": symbol, "action": "OPEN", "side": side,
        "entry": price, "exit": "", "qty": qty,
        "sl": sl_price, "tp1": tp1_price, "tp2": tp2_price,
        "pnl_pct": "", "pnl_usdt": "", "reason": "SIGNAL",
    }
    log_trade(row)
    trade_log.append({**row, "price": price})
    if len(trade_log) > 20:
        trade_log.pop(0)

    notify(
        f"🟢 OPEN {side} {symbol} @ ${price:,.4f}\n"
        f"qty {qty} | SL ${sl_price:,.4f} | TP1 ${tp1_price:,.4f} | TP2 ${tp2_price:,.4f}",
        "TRADE",
    )
    return True


async def close_position(
    client: AsyncClient, symbol: str,
    reason: str = "SIGNAL", partial: bool = False, partial_qty: float = 0.0,
):
    global _daily_realized_pnl, _cached_balance_ts

    pos  = market_data[symbol]["position"]
    side = pos["side"]
    qty  = partial_qty if partial else pos["qty"]
    if side is None or qty == 0:
        return

    if not partial:
        await cancel_open_orders(client, symbol)

    close_side = SIDE_SELL if side == "LONG" else SIDE_BUY
    ok = await _place_market(client, symbol, close_side, qty)
    if not ok:
        return

    pnl_usdt = pos["pnl_usdt"] * (qty / pos["qty"]) if pos["qty"] > 0 else 0.0
    _daily_realized_pnl += pnl_usdt
    _cached_balance_ts   = 0.0   # force balance refresh after close

    pnl_col = "green" if pnl_usdt >= 0 else "red"
    console.print(
        f"[cyan]🔒 {'PARTIAL ' if partial else ''}Close {side} {symbol} "
        f"| {reason} | PnL: [{pnl_col}]{pos['pnl_pct']:.2f}%[/][/cyan]"
    )

    row = {
        "date": date.today().isoformat(), "time": datetime.now().strftime("%H:%M:%S"),
        "symbol": symbol, "action": "PARTIAL_CLOSE" if partial else "CLOSE",
        "side": side, "entry": pos["entry"], "exit": market_data[symbol]["price"],
        "qty": qty, "sl": pos["sl_price"], "tp1": pos["tp1_price"], "tp2": pos["tp2_price"],
        "pnl_pct": f"{pos['pnl_pct']:.2f}%", "pnl_usdt": f"{pnl_usdt:.2f}", "reason": reason,
    }
    log_trade(row)
    trade_log.append({**row, "price": market_data[symbol]["price"], "pnl": f"{pos['pnl_pct']:+.2f}%"})
    if len(trade_log) > 20:
        trade_log.pop(0)

    kind = "🟠 PARTIAL CLOSE" if partial else "🔴 CLOSE"
    notify(
        f"{kind} {side} {symbol} @ ${market_data[symbol]['price']:,.4f}\n"
        f"{reason} | PnL {pos['pnl_pct']:+.2f}% (${pnl_usdt:+.2f})",
        "TRADE",
    )
    if partial:
        pos["qty"]      -= qty
        pos["tp1_hit"]   = True
        pos["sl_price"]  = pos["entry"]   # move SL to breakeven (software)
        # Move the EXCHANGE stop to breakeven too, so protection survives a crash
        new_sl_id = await replace_exchange_sl(client, symbol, side, pos["entry"])
        pos["sl_order_id"] = new_sl_id
        console.print(f"[cyan]  → SL moved to breakeven @ ${pos['entry']:,.4f}[/cyan]")
    else:
        if pnl_usdt < 0:
            cooldown_until[symbol] = time.time() + COOLDOWN_AFTER_LOSS
            console.print(f"[yellow]⏳ {symbol} cooldown {COOLDOWN_AFTER_LOSS}s after loss[/yellow]")
        market_data[symbol]["position"] = _empty_position()

# ╔══════════════════════════════════════════════════════════════════╗
# ║                       AI FUNCTIONS                             ║
# ╚══════════════════════════════════════════════════════════════════╝

def _sentiment_blocking(symbol: str) -> str:
    """Blocking sentiment call — run inside executor."""
    try:
        resp = AI_client.chat.completions.create(
            model=os.getenv("MODEL", "grok-beta"),
            messages=[{"role": "user", "content": (
                f"You are a market sentiment classifier.\n"
                f"Determine CURRENT sentiment for {symbol} crypto futures.\n"
                f"Output EXACTLY ONE word: BULLISH, BEARISH, or NEUTRAL.\n"
                f"No punctuation, no explanation."
            )}],
            temperature=0.0, max_tokens=10,
        )
        sent = resp.choices[0].message.content.strip().upper()
        return sent if sent in ("BULLISH", "BEARISH", "NEUTRAL") else "NEUTRAL"
    except Exception:
        return "NEUTRAL"


async def get_AI_sentiment(symbol: str) -> str:
    """Non-blocking wrapper — runs the OpenAI call in a thread."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _sentiment_blocking, symbol)


def _analysis_blocking(symbol: str, price: float) -> str:
    """Blocking deep-analysis call — run inside executor."""
    d   = market_data[symbol]
    pos = d["position"]
    pos_str = (
        f"OPEN {pos['side']} @ ${pos['entry']:,.4f} | "
        f"PnL: {pos['pnl_pct']:+.2f}% | SL: ${pos['sl_price']:,.4f} | "
        f"TP1: ${pos['tp1_price']:,.4f} | TP2: ${pos['tp2_price']:,.4f}"
        if pos["side"] else "No open position"
    )
    try:
        prompt = f"""You are a senior crypto futures prop trader. Be concise and direct.

Symbol: {symbol}
Price:  ${price:,.4f}
RSI 15m / 1h: {d['rsi_15']:.1f} / {d['rsi_1h']:.1f}
SMA20: {d['sma20']:.4f}  EMA50: {d['ema50']:.4f}  EMA200: {d['ema200']:.4f}
MACD: {d['macd']:.4f} | Signal: {d['macd_sig']:.4f}
ATR(14): {d['atr']:.4f}  ADX: {d['adx']:.1f}
BB Squeeze (low vol breakout setup): {d['bb_squeeze']}
Volume above avg: {d['volume_ok']}
AI Sentiment: {d['sentiment']}
Current Position: {pos_str}

Respond EXACTLY in this format — fill in every field:

**OVERALL BIAS:** BULLISH / BEARISH / NEUTRAL
**CONFIDENCE:** XX/100
**MARKET SNAPSHOT:**
- Price Action: 
- Key Support/Resistance: 
- Volume Trend: 
- EMA Alignment: 
- MACD: 
- ADX / Regime: 
**MULTI-TIMEFRAME ANALYSIS:**
- 15m: 
- 1h: 
**TRADING RECOMMENDATION:**
- Action: LONG / SHORT / WAIT / CLOSE
- Entry Zone: 
- Stop Loss: 
- Take Profit 1: 
- Take Profit 2: 
**REASONING:** 2-3 sentences only.
"""
        resp = AI_client.chat.completions.create(
            model=os.getenv("MODEL", "grok-beta"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=700,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Analysis Error: {str(e)[:150]}"


async def get_deep_analysis(symbol: str, price: float) -> str:
    """Non-blocking wrapper."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _analysis_blocking, symbol, price)

# ╔══════════════════════════════════════════════════════════════════╗
# ║                       INDICATORS                               ║
# ╚══════════════════════════════════════════════════════════════════╝

async def fetch_df(
    client: AsyncClient, symbol: str, interval: str, limit: int = 250
) -> pd.DataFrame | None:
    try:
        klines = await client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            "open_time","Open","High","Low","Close","Volume",
            "close_time","qav","n_trades","taker_base","taker_quote","ignore",
        ]).astype(float)
        return df
    except Exception as e:
        console.print(f"[yellow]⚠ fetch_df {symbol} {interval}: {e}[/yellow]")
        return None


def _bb_squeeze(df: pd.DataFrame, length: int = SQUEEZE_LOOKBACK, std: float = BB_STD) -> bool:
    """
    True when current Bollinger bandwidth is in the lower 20th percentile
    (a 'squeeze' / low-volatility coil). Robust to pandas_ta column-name
    variations: the bandwidth column may be 'BBB_20_2.0', 'BBB_20_2', etc.
    """
    try:
        bb = ta.bbands(df["Close"], length=length, std=std)
        if bb is None or bb.empty:
            return False
        # Find the bandwidth column (starts with 'BBB') regardless of suffix format
        bbb_cols = [c for c in bb.columns if c.upper().startswith("BBB")]
        if bbb_cols:
            width = bb[bbb_cols[0]]
        else:
            # Fallback: compute bandwidth from upper/lower/mid bands
            upper = next((c for c in bb.columns if c.upper().startswith("BBU")), None)
            lower = next((c for c in bb.columns if c.upper().startswith("BBL")), None)
            mid   = next((c for c in bb.columns if c.upper().startswith("BBM")), None)
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


async def update_one_symbol(client: AsyncClient, symbol: str):
    """
    Refresh indicators for a SINGLE symbol.
    Both timeframes are fetched concurrently with asyncio.gather.
    The WS-updated price is NOT overwritten unless WS data is stale.
    """
    df15, df1h = await asyncio.gather(
        fetch_df(client, symbol, TF_PRIMARY, 250),
        fetch_df(client, symbol, TF_CONFIRM,  60),
    )
    if df15 is None or len(df15) < 60:
        return

    df15["RSI"]    = ta.rsi(df15["Close"],  length=14)
    df15["SMA20"]  = ta.sma(df15["Close"],  length=20)
    df15["EMA50"]  = ta.ema(df15["Close"],  length=50)
    df15["EMA200"] = ta.ema(df15["Close"],  length=200)
    df15["ATR"]    = ta.atr(df15["High"], df15["Low"], df15["Close"], length=14)
    adx_df         = ta.adx(df15["High"], df15["Low"], df15["Close"], length=14)
    macd_df        = ta.macd(df15["Close"], fast=12, slow=26, signal=9)
    vol_avg        = df15["Volume"].rolling(20).mean()

    last = df15.iloc[-1]
    d    = market_data[symbol]

    def _f(val, fallback):
        return float(val) if not pd.isna(val) else fallback

    # Only overwrite price if the WS price is stale (no recent tick)
    ws_fresh = time.time() - d["price_ts"] < STALE_PRICE_THRESHOLD
    if not ws_fresh:
        d["price"]    = _f(last["Close"], d["price"])
        d["price_ts"] = time.time()

    d["sma20"]     = _f(last["SMA20"],  d["sma20"])
    d["ema50"]     = _f(last["EMA50"],  d["ema50"])
    d["ema200"]    = _f(last["EMA200"], d["ema200"])
    d["rsi_15"]    = _f(last["RSI"],    d["rsi_15"])
    d["atr"]       = _f(last["ATR"],    d["atr"])
    d["volume_ok"] = (
        float(last["Volume"]) > float(vol_avg.iloc[-1])
        if not pd.isna(vol_avg.iloc[-1]) else False
    )
    d["ind_ts"] = time.time()

    if adx_df is not None and "ADX_14" in adx_df.columns:
        d["adx"] = _f(adx_df["ADX_14"].iloc[-1], d["adx"])

    if macd_df is not None:
        d["macd"]     = _f(macd_df["MACD_12_26_9"].iloc[-1],  d["macd"])
        d["macd_sig"] = _f(macd_df["MACDs_12_26_9"].iloc[-1], d["macd_sig"])

    d["bb_squeeze"] = _bb_squeeze(df15)

    if df1h is not None and len(df1h) > 14:
        df1h["RSI1H"] = ta.rsi(df1h["Close"], length=14)
        d["rsi_1h"]   = _f(df1h["RSI1H"].iloc[-1], d["rsi_1h"])

# ╔══════════════════════════════════════════════════════════════════╗
# ║                      TRADING LOGIC                             ║
# ╚══════════════════════════════════════════════════════════════════╝

def compute_signal(symbol: str) -> str:
    d   = market_data[symbol]
    pos = d["position"]
    p   = d["price"]

    trending  = d["adx"] >= ADX_MIN
    macd_bull = d["macd"] > d["macd_sig"]
    macd_bear = d["macd"] < d["macd_sig"]
    vol_ok    = d["volume_ok"]

    if pos["side"] is None:
        # ── Long entry gates ─────────────────────────────────────
        long_ema  = p > d["ema50"] > d["ema200"]
        long_rsi  = 38 <= d["rsi_15"] <= 65 and d["rsi_1h"] < 70
        long_sent = d["sentiment"] != "BEARISH"
        long_ok   = long_ema and long_rsi and macd_bull and vol_ok and trending and long_sent

        # ── Short entry gates ────────────────────────────────────
        short_ema  = p < d["ema50"] < d["ema200"]
        short_rsi  = 35 <= d["rsi_15"] <= 62 and d["rsi_1h"] > 30
        short_sent = d["sentiment"] != "BULLISH"
        short_ok   = short_ema and short_rsi and macd_bear and vol_ok and trending and short_sent

        if long_ok:
            d["hold_reason"] = ""
            return "BUY"
        if short_ok:
            d["hold_reason"] = ""
            return "SHORT"

        # ── No entry: record WHY (whichever side is closer to firing) ──
        # Pick the side whose trend structure already aligns; default to long.
        if short_ema and not long_ema:
            # market structure is bearish → explain short-side block
            if   not short_rsi:  d["hold_reason"] = "rsi"
            elif not macd_bear:  d["hold_reason"] = "macd"
            elif not vol_ok:     d["hold_reason"] = "vol"
            elif not trending:   d["hold_reason"] = f"adx{d['adx']:.0f}<{ADX_MIN:.0f}"
            elif not short_sent: d["hold_reason"] = "sent"
            else:                d["hold_reason"] = "ema"
        else:
            # explain long-side block
            if   not long_ema:   d["hold_reason"] = "ema"
            elif not long_rsi:   d["hold_reason"] = "rsi"
            elif not macd_bull:  d["hold_reason"] = "macd"
            elif not vol_ok:     d["hold_reason"] = "vol"
            elif not trending:   d["hold_reason"] = f"adx{d['adx']:.0f}<{ADX_MIN:.0f}"
            elif not long_sent:  d["hold_reason"] = "sent"
            else:                d["hold_reason"] = "—"

    if pos["side"] == "LONG":
        d["hold_reason"] = ""
        if (p < d["ema50"] or d["rsi_15"] > 75 or macd_bear or
                (d["sentiment"] == "BEARISH" and p < d["sma20"])):
            return "SELL_LONG"

    if pos["side"] == "SHORT":
        d["hold_reason"] = ""
        if (p > d["ema50"] or d["rsi_15"] < 25 or macd_bull or
                (d["sentiment"] == "BULLISH" and p > d["sma20"])):
            return "COVER_SHORT"

    return "HOLD"


def count_open_longs() -> int:
    return sum(1 for s in SYMBOLS if market_data[s]["position"]["side"] == "LONG")

def count_open_shorts() -> int:
    return sum(1 for s in SYMBOLS if market_data[s]["position"]["side"] == "SHORT")

def count_open_positions() -> int:
    return count_open_longs() + count_open_shorts()


def reshuffle_symbols():
    """
    Pick a fresh random set of ACTIVE_SYMBOL_COUNT symbols from the pool.
    Symbols with open positions are always kept; the rest are replaced.
    Called every SYMBOL_RESHUFFLE_HOURS hours.
    """
    global SYMBOLS, MAIN_SYMBOL, _indicator_index, _ws_needs_restart

    # Keep any symbol that currently has an open position
    pinned = [s for s in SYMBOLS if market_data[s]["position"]["side"] is not None]
    # Fill remaining slots from pool, excluding pinned symbols
    pool_rest = [s for s in _SYMBOL_POOL if s not in pinned]
    random.shuffle(pool_rest)
    slots = max(0, ACTIVE_SYMBOL_COUNT - len(pinned))
    new_symbols = pinned + pool_rest[:slots]
    random.shuffle(new_symbols)   # mix pinned and new together

    # Add market_data entries for any brand-new symbol
    for sym in new_symbols:
        if sym not in market_data:
            market_data[sym] = {
                "price": 0.0, "price_ts": 0.0,
                "rsi_15": 50.0, "rsi_1h": 50.0,
                "sma20": 0.0, "ema50": 0.0, "ema200": 0.0,
                "macd": 0.0, "macd_sig": 0.0,
                "atr": 0.0, "adx": 0.0,
                "bb_squeeze": False, "volume_ok": False,
                "signal": "HOLD", "hold_reason": "", "sentiment": "NEUTRAL",
                "ind_ts": 0.0,
                "position": _empty_position(),
            }
            analyses[sym]      = "🔄 Loading…"
            analysis_time[sym] = 0.0
            cooldown_until[sym]= 0.0

    removed = [s for s in SYMBOLS if s not in new_symbols]
    added   = [s for s in new_symbols if s not in SYMBOLS]
    SYMBOLS[:] = new_symbols
    MAIN_SYMBOL = SYMBOLS[0]
    _indicator_index = 0   # restart rotation from beginning of new list

    if added or removed:
        _ws_needs_restart = True   # force WS to re-subscribe to the new symbol set
        console.print(
            f"[magenta]🔀 Symbol reshuffle | "
            f"+{len(added)} {added[:3]}{'…' if len(added)>3 else ''} | "
            f"-{len(removed)} {removed[:3]}{'…' if len(removed)>3 else ''}[/magenta]"
        )
        notify(f"🔀 Symbol reshuffle\n+{added}\n-{removed}", "INFO")


def _update_trailing_stop(symbol: str):
    pos  = market_data[symbol]["position"]
    side = pos["side"]
    if side is None:
        return
    price        = market_data[symbol]["price"]
    entry        = pos["entry"]
    sl_dist      = abs(entry - pos["sl_price"])
    if sl_dist == 0:
        return
    activation   = sl_dist * TRAIL_ACTIVATION_MULT
    trail_offset = sl_dist * TRAIL_OFFSET_MULT
    profit = (price - entry) if side == "LONG" else (entry - price)

    if not pos["trail_active"]:
        if profit >= activation:
            pos["trail_active"] = True
            pos["trail_best"]   = price
            pos["trail_stop"]   = (
                price - trail_offset if side == "LONG" else price + trail_offset
            )
    else:
        if side == "LONG" and price > pos["trail_best"]:
            pos["trail_best"]  = price
            pos["trail_stop"]  = max(pos["trail_stop"], price - trail_offset)
        elif side == "SHORT" and price < pos["trail_best"]:
            pos["trail_best"]  = price
            pos["trail_stop"]  = min(pos["trail_stop"], price + trail_offset)


async def run_trading_logic(client: AsyncClient, balance: float):
    global _trading_halted, _last_reshuffle

    # ── Symbol reshuffle check ────────────────────────────────────
    if time.time() - _last_reshuffle >= SYMBOL_RESHUFFLE_HOURS * 3600:
        reshuffle_symbols()
        _last_reshuffle = time.time()

    # ── Daily loss circuit breaker ────────────────────────────────
    if _session_start_balance > 0:
        daily_loss_pct = _daily_realized_pnl / _session_start_balance
        if daily_loss_pct <= -DAILY_LOSS_LIMIT_PCT:
            if not _trading_halted:
                console.print(
                    f"[bold red]🚨 Daily loss limit hit "
                    f"({daily_loss_pct*100:.1f}%). Halting new entries.[/bold red]"
                )
                notify(f"🚨 Daily loss limit hit ({daily_loss_pct*100:.1f}%). "
                       f"Halting new entries.", "RISK")
            _trading_halted = True
        else:
            _trading_halted = False

    open_longs  = count_open_longs()
    open_shorts = count_open_shorts()

    for symbol in SYMBOLS:
        d   = market_data[symbol]
        pos = d["position"]
        p   = d["price"]

        # Skip stale prices
        if d["price_ts"] > 0 and time.time() - d["price_ts"] > STALE_PRICE_THRESHOLD:
            continue

        _update_trailing_stop(symbol)

        # ── Software SL / TP checks ───────────────────────────────
        if pos["side"] == "LONG" and pos["entry"] > 0:
            trail_breach = pos["trail_active"] and p <= pos["trail_stop"]
            sl_breach    = not pos["trail_active"] and p <= pos["sl_price"]
            if trail_breach or sl_breach:
                await close_position(client, symbol, reason="TRAIL-SL" if trail_breach else "SL-SW")
                open_longs = count_open_longs()
                continue
            if not pos["tp1_hit"] and p >= pos["tp1_price"]:
                half = _round_step(pos["qty"] * PARTIAL_CLOSE_FRACTION, 0.001)
                await close_position(client, symbol, reason="TP1-PARTIAL", partial=True, partial_qty=half)
            elif p >= pos["tp2_price"]:
                await close_position(client, symbol, reason="TP2")
                open_longs = count_open_longs()
                continue

        if pos["side"] == "SHORT" and pos["entry"] > 0:
            trail_breach = pos["trail_active"] and p >= pos["trail_stop"]
            sl_breach    = not pos["trail_active"] and p >= pos["sl_price"]
            if trail_breach or sl_breach:
                await close_position(client, symbol, reason="TRAIL-SL" if trail_breach else "SL-SW")
                open_shorts = count_open_shorts()
                continue
            if not pos["tp1_hit"] and p <= pos["tp1_price"]:
                half = _round_step(pos["qty"] * PARTIAL_CLOSE_FRACTION, 0.001)
                await close_position(client, symbol, reason="TP1-PARTIAL", partial=True, partial_qty=half)
            elif p <= pos["tp2_price"]:
                await close_position(client, symbol, reason="TP2")
                open_shorts = count_open_shorts()
                continue

        # ── Signal ────────────────────────────────────────────────
        signal      = compute_signal(symbol)
        d["signal"] = signal

        # ── Signal-flip: close opposite side then open new ────────
        flipping = False
        if signal == "BUY" and pos["side"] == "SHORT":
            console.print(f"[yellow]↩ Signal flip {symbol}: closing SHORT → opening LONG[/yellow]")
            await close_position(client, symbol, reason="FLIP-TO-LONG")
            open_shorts = count_open_shorts()
            pos = market_data[symbol]["position"]   # re-read after close
            flipping = True

        elif signal == "SHORT" and pos["side"] == "LONG":
            console.print(f"[yellow]↩ Signal flip {symbol}: closing LONG → opening SHORT[/yellow]")
            await close_position(client, symbol, reason="FLIP-TO-SHORT")
            open_longs = count_open_longs()
            pos = market_data[symbol]["position"]
            flipping = True

        # A flip just closed at a loss would set a cooldown that blocks its own
        # reopen — clear it so the flip can complete this tick.
        if flipping:
            cooldown_until[symbol] = 0.0

        # ── Open new LONG ─────────────────────────────────────────
        if signal == "BUY" and pos["side"] is None:
            if _trading_halted or balance < MIN_USDT_TO_TRADE:
                continue
            if open_longs >= MAX_OPEN_LONGS:
                d["signal"] = "HOLD"   # show as hold, slot full
                continue
            if not flipping and time.time() < cooldown_until[symbol]:
                continue
            if await open_position(client, symbol, "LONG", balance):
                open_longs += 1

        # ── Open new SHORT ────────────────────────────────────────
        elif signal == "SHORT" and pos["side"] is None:
            if _trading_halted or balance < MIN_USDT_TO_TRADE:
                continue
            if open_shorts >= MAX_OPEN_SHORTS:
                d["signal"] = "HOLD"
                continue
            if not flipping and time.time() < cooldown_until[symbol]:
                continue
            if await open_position(client, symbol, "SHORT", balance):
                open_shorts += 1

        # ── Close signals (non-flip) ──────────────────────────────
        elif signal == "SELL_LONG" and pos["side"] == "LONG":
            await close_position(client, symbol, reason="SIGNAL")
            open_longs = count_open_longs()

        elif signal == "COVER_SHORT" and pos["side"] == "SHORT":
            await close_position(client, symbol, reason="SIGNAL")
            open_shorts = count_open_shorts()

# ╔══════════════════════════════════════════════════════════════════╗
# ║                         DISPLAY                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def _term_size() -> tuple[int, int]:
    try:
        s = os.get_terminal_size()
        return s.columns, s.lines
    except OSError:
        return 160, 45


def _extract(text: str, key: str, default: str = "—") -> str:
    m = re.search(rf"\*\*{re.escape(key)}[:\*]*\*?\*?\s*(.+)", text)
    return m.group(1).strip().strip("*").strip() if m else default


def _extract_block(text: str, key: str) -> list[str]:
    lines, capturing = [], False
    for line in text.splitlines():
        if re.search(rf"\*\*{re.escape(key)}", line, re.IGNORECASE):
            capturing = True
            continue
        if capturing:
            if line.strip().startswith("**") and ":" in line:
                break
            if line.strip().startswith("-"):
                lines.append(line.strip().lstrip("- ").strip())
            elif line.strip() and not line.strip().startswith("**"):
                lines.append(line.strip())
    return [l for l in lines if l]


def _wrap(text: str, width: int = 46) -> list[str]:
    words, buf, out = text.split(), [], []
    for w in words:
        buf.append(w)
        if len(" ".join(buf)) > width:
            out.append(" ".join(buf[:-1]))
            buf = [w]
    if buf:
        out.append(" ".join(buf))
    return out


def render_analysis_card(sym: str) -> Table:
    raw = analyses[sym]
    d   = market_data[sym]
    pos = d["position"]

    bias       = _extract(raw, "OVERALL BIAS")
    confidence = _extract(raw, "CONFIDENCE")
    action     = _extract(raw, "Action")
    entry_z    = _extract(raw, "Entry Zone")
    sl         = _extract(raw, "Stop Loss")
    tp1        = _extract(raw, "Take Profit 1")
    tp2        = _extract(raw, "Take Profit 2")
    reasoning  = _extract(raw, "REASONING")
    snap_lines = _extract_block(raw, "MARKET SNAPSHOT")
    mtf_lines  = _extract_block(raw, "MULTI-TIMEFRAME")

    bias_col   = {"BULLISH": "green", "BEARISH": "red"}.get(bias.upper().split()[0], "yellow")
    action_col = {"LONG": "green", "SHORT": "red", "CLOSE": "red", "WAIT": "yellow"}.get(
        action.upper().split()[0], "white")
    age_s   = int(time.time() - analysis_time[sym])
    age_col = "green" if age_s < 120 else "yellow" if age_s < 300 else "red"
    ind_age = int(time.time() - d["ind_ts"]) if d["ind_ts"] > 0 else -1

    card = Table.grid(padding=(0, 1))
    card.add_column(style="dim", width=14, no_wrap=True)
    card.add_column(overflow="fold")

    def row(label, value, style=""):
        v = f"[{style}]{value}[/]" if style else str(value)
        card.add_row(f"[dim]{label}[/dim]", v)

    def sep(title=""):
        line = f"── {title} " + "─" * max(2, 20-len(title)) if title else "─" * 22
        card.add_row("", f"[dim]{line}[/dim]")

    row("Price",       f"${d['price']:,.4f}", "bold white")
    ind_str = f"ind {ind_age}s ago" if ind_age >= 0 else "pending"
    row("Data age",    ind_str, "green" if ind_age < 60 else "yellow" if ind_age < 120 else "red")
    row("RSI 15m/1h",  f"{d['rsi_15']:.1f} / {d['rsi_1h']:.1f}")
    row("SMA20/EMA50", f"{d['sma20']:,.2f} / {d['ema50']:,.2f}")
    row("EMA200",      f"{d['ema200']:,.2f}")
    row("MACD/Sig",    f"{d['macd']:.3f} / {d['macd_sig']:.3f}")
    row("ATR / ADX",   f"{d['atr']:.4f} / {d['adx']:.1f}")
    row("BB Squeeze",  "✓" if d["bb_squeeze"] else "✗",
        "cyan" if d["bb_squeeze"] else "dim")
    row("Vol > Avg",   "✓" if d["volume_ok"] else "✗",
        "green" if d["volume_ok"] else "red")
    row("Sentiment",   d["sentiment"],
        "green" if d["sentiment"]=="BULLISH" else "red" if d["sentiment"]=="BEARISH" else "yellow")

    sep("Bias")
    row("Overall",    bias,       bias_col)
    row("Confidence", confidence, bias_col)

    if snap_lines:
        sep("Snapshot")
        for l in snap_lines[:5]:
            card.add_row("", f"[dim]• {l}[/dim]")

    if mtf_lines:
        sep("Multi-TF")
        for l in mtf_lines[:3]:
            card.add_row("", f"[dim]• {l}[/dim]")

    sep("Trade")
    live_sig = d["signal"]
    if live_sig == "HOLD" and d.get("hold_reason"):
        sig_disp = f"HOLD (blocked: {d['hold_reason']})"
        sig_col_c = "yellow"
    else:
        sig_disp  = live_sig
        sig_col_c = {"BUY": "green", "SHORT": "red",
                     "SELL_LONG": "red", "COVER_SHORT": "green"}.get(live_sig, "dim")
    row("Live Signal", sig_disp, sig_col_c)
    row("Action",     action, action_col)
    row("Entry Zone", entry_z)
    row("Stop Loss",  sl,  "red")
    row("TP1",        tp1, "green")
    row("TP2",        tp2, "green")

    if pos["side"]:
        pnl_col = "green" if pos["pnl_pct"] >= 0 else "red"
        sep("Open Position")
        row("Side",    pos["side"], "cyan")
        row("Entry",   f"${pos['entry']:,.4f}")
        row("SL/TP1",  f"${pos['sl_price']:,.4f} / ${pos['tp1_price']:,.4f}")
        if pos["trail_active"]:
            row("Trail Stop", f"${pos['trail_stop']:,.4f}", "magenta")
        row("TP1 Hit", "✓" if pos["tp1_hit"] else "✗",
            "green" if pos["tp1_hit"] else "dim")
        row("PnL",     f"{pos['pnl_pct']:+.2f}%  (${pos['pnl_usdt']:+.2f})", pnl_col)

    if reasoning and reasoning != "—":
        sep("Reasoning")
        for wl in _wrap(reasoning, 45):
            card.add_row("", f"[italic dim]{wl}[/italic dim]")

    card.add_row("", f"[{age_col}][dim]AI {age_s}s ago[/dim][/]")
    return card


def build_summary_table(balance: float, width: int) -> Table:
    sig_col  = {"BUY": "bold green", "SHORT": "bold red", "HOLD": "dim white",
                "SELL_LONG": "red", "COVER_SHORT": "green"}
    sent_col = {"BULLISH": "green", "BEARISH": "red", "NEUTRAL": "yellow"}
    ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    daily    = _daily_realized_pnl
    daily_s  = f"[green]+${daily:.2f}[/]" if daily >= 0 else f"[red]-${abs(daily):.2f}[/]"
    halted   = "  [bold red]⛔ HALTED[/bold red]" if _trading_halted else ""
    n_long   = count_open_longs()
    n_short  = count_open_shorts()
    reshuffle_in = max(0, SYMBOL_RESHUFFLE_HOURS * 3600 - (time.time() - _last_reshuffle))
    reshuffle_m  = int(reshuffle_in / 60)

    table = Table(
        title=(
            f"[bold]⚡ AI Futures Trader v4[/bold]  "
            f"💰 ${balance:,.2f} USDT  "
            f"📈 Daily PnL: {daily_s}  "
            f"🕐 {ts}  📊 {LEVERAGE}x  "
            f"[green]L:{n_long}/{MAX_OPEN_LONGS}[/]  "
            f"[red]S:{n_short}/{MAX_OPEN_SHORTS}[/]  "
            f"[dim]🔀 reshuffle {reshuffle_m}m[/dim]{halted}"
        ),
        show_lines=True, expand=True, border_style="dim",
    )
    table.add_column("Symbol",   style="bold cyan", justify="center", min_width=12)
    table.add_column("Price",    justify="right",   min_width=12)
    table.add_column("RSI 15m",  justify="right",   min_width=7)
    table.add_column("ADX",      justify="right",   min_width=6)
    table.add_column("Signal",   justify="center",  min_width=15)
    table.add_column("Position", justify="center",  min_width=7)
    table.add_column("PnL %",    justify="right",   min_width=8)
    table.add_column("PnL USDT", justify="right",   min_width=9)
    if width >= 130:
        table.add_column("Sent",  justify="center", min_width=9)
        table.add_column("Entry", justify="right",  min_width=12)
        table.add_column("SL",    justify="right",  min_width=10)
        table.add_column("TP1",   justify="right",  min_width=10)
    if width >= 175:
        table.add_column("Trail", justify="center", min_width=5)
        table.add_column("Sqz",   justify="center", min_width=4)
        table.add_column("Vol✓",  justify="center", min_width=4)
        table.add_column("Ind",   justify="right",  min_width=6)  # data freshness

    for sym in SYMBOLS:
        d   = market_data[sym]
        pos = d["position"]

        pnl_pct_str = "[dim]—[/dim]"
        pnl_usd_str = "[dim]—[/dim]"
        entry_str   = "[dim]—[/dim]"
        sl_str      = "[dim]—[/dim]"
        tp1_str     = "[dim]—[/dim]"

        if pos["side"]:
            c = "green" if pos["pnl_pct"] >= 0 else "red"
            pnl_pct_str = f"[{c}]{pos['pnl_pct']:+.2f}%[/]"
            pnl_usd_str = f"[{c}]{pos['pnl_usdt']:+.2f}[/]"
            entry_str   = f"${pos['entry']:,.4f}"
            sl_str      = f"[red]${pos['sl_price']:,.4f}[/red]"
            tp1_str     = f"[green]${pos['tp1_price']:,.4f}[/green]"

        rsi   = d["rsi_15"]
        adx   = d["adx"]
        rsi_c = "green" if 40<=rsi<=60 else "red" if rsi>70 or rsi<30 else "yellow"
        adx_c = "green" if adx >= ADX_MIN else "red"
        sc    = sig_col.get(d["signal"], "white")
        se    = sent_col.get(d["sentiment"], "yellow")
        pos_s = pos["side"] or "[dim]—[/dim]"
        if pos["side"] and pos["trail_active"]:
            pos_s = f"[magenta]{pos['side']}~[/magenta]"

        # Signal cell — append the blocking gate when HOLD has a reason
        if d["signal"] == "HOLD" and d.get("hold_reason"):
            signal_cell = f"[{sc}]HOLD[/] [dim]{d['hold_reason']}[/dim]"
        else:
            signal_cell = f"[{sc}]{d['signal']}[/]"

        # Indicator freshness colour
        ind_age = time.time() - d["ind_ts"] if d["ind_ts"] > 0 else 9999
        ind_c   = "green" if ind_age < 60 else "yellow" if ind_age < 200 else "red"
        ind_str = f"[{ind_c}]{int(ind_age)}s[/]" if d["ind_ts"] > 0 else "[dim]—[/dim]"

        row_data = [
            sym,
            f"[bold]${d['price']:,.4f}[/bold]",
            f"[{rsi_c}]{rsi:.1f}[/]",
            f"[{adx_c}]{adx:.0f}[/]",
            signal_cell,
            pos_s,
            pnl_pct_str,
            pnl_usd_str,
        ]
        if width >= 130:
            row_data += [f"[{se}]{d['sentiment']}[/]", entry_str, sl_str, tp1_str]
        if width >= 175:
            row_data += [
                "🟣" if pos.get("trail_active") else "·",
                "🔵" if d["bb_squeeze"] else "·",
                "✓" if d["volume_ok"] else "·",
                ind_str,
            ]
        table.add_row(*row_data)

    return table


def build_trade_log_panel() -> Panel:
    if not trade_log:
        return Panel("[dim]No trades yet.[/dim]", title="📋 Trade Log", border_style="dim")
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", width=8)
    grid.add_column(style="bold cyan", width=14)
    grid.add_column(width=22)
    grid.add_column(justify="right", width=12)
    grid.add_column(justify="right", width=9)
    for t in reversed(trade_log[-8:]):
        action = t.get("action", "")
        col    = ("green" if "OPEN" in action and t.get("side") == "LONG" else
                  "red"   if "OPEN" in action and t.get("side") == "SHORT" else
                  "cyan"  if "CLOSE" in action else "white")
        grid.add_row(
            t["time"],
            t["symbol"],
            f"[{col}]{action} {t.get('side','')}[/]",
            f"${t.get('price', t.get('exit', 0)):,.4f}",
            t.get("pnl", ""),
        )
    return Panel(grid, title="📋 Trade Log", border_style="dim")


def build_analysis_section(width: int):
    panels = [
        Panel(
            render_analysis_card(sym),
            title=f"[bold]🧠 {sym}[/bold]",
            border_style="yellow" if sym == MAIN_SYMBOL else "blue",
            expand=True, padding=(0, 1),
        )
        for sym in SYMBOLS
    ]
    n = len(panels)
    if n == 0:      return Panel("[dim]No symbols.[/dim]")
    if width < 100: return panels[0]
    cols = 4 if width >= 260 else 3 if width >= 200 else 2
    rows = [Columns(panels[i:i+cols], equal=True, expand=True) for i in range(0, n, cols)]
    return RichGroup(*rows)


def build_layout(balance: float) -> Layout:
    width, height = _term_size()
    table_h = len(SYMBOLS) + 7
    log_h   = 12

    layout = Layout()
    if height < table_h + log_h + 10:   # ← fixed: was +100
        layout.split_column(Layout(name="table", ratio=1))
        layout["table"].update(Panel(build_summary_table(balance, width), border_style="dim"))
        return layout

    layout.split_column(
        Layout(name="table", size=table_h),
        Layout(name="main",  ratio=1),
        Layout(name="log",   size=log_h),
    )
    layout["table"].update(
        Panel(build_summary_table(balance, width), border_style="dim", padding=(0, 1))
    )
    layout["main"].update(
        Panel(
            build_analysis_section(width),
            title="[bold]📊 AI Analysis[/bold]",
            border_style="dim", padding=(0, 1),
        )
    )
    layout["log"].update(build_trade_log_panel())
    return layout

# ╔══════════════════════════════════════════════════════════════════╗
# ║                        MAIN LOOP                               ║
# ╚══════════════════════════════════════════════════════════════════╝

class _NoopLive:
    """Stand-in for rich.Live when running headless (Docker/piped output).
    Has the same `.update()` method so the loop code doesn't change."""
    def __enter__(self):  return self
    def __exit__(self, *a): return False
    def update(self, *a, **k): pass


def print_plain_status(balance: float):
    """One concise line per symbol — readable in `docker compose logs`."""
    n_long, n_short = count_open_longs(), count_open_shorts()
    halt = " HALTED" if _trading_halted else ""
    console.print(
        f"[bold]─ {datetime.now():%H:%M:%S} ─[/bold] "
        f"💰 ${balance:,.2f} | PnL ${_daily_realized_pnl:+.2f} | "
        f"L:{n_long}/{MAX_OPEN_LONGS} S:{n_short}/{MAX_OPEN_SHORTS}{halt}"
    )
    for sym in SYMBOLS:
        d   = market_data[sym]
        pos = d["position"]
        sig = d["signal"]
        tag = f" ({d.get('hold_reason','')})" if sig == "HOLD" and d.get("hold_reason") else ""
        if pos["side"]:
            posinfo = f"{pos['side']} {pos['pnl_pct']:+.2f}% (${pos['pnl_usdt']:+.2f})"
        else:
            posinfo = "—"
        price = d["price"]
        pstr  = f"${price:,.4f}" if price else "  (loading)"
        console.print(
            f"  {sym:<13} {pstr:>14}  RSI {d['rsi_15']:5.1f}  "
            f"ADX {d['adx']:4.0f}  {sig}{tag:<10}  {posinfo}"
        )


async def trading_loop(client: AsyncClient, live, state: dict):
    global _indicator_index, _last_indicator_tick

    now = time.time()

    # ── Position sync (every 8 s) ────────────────────────────────
    if now - state["last_position_sync"] >= POSITION_SYNC_INTERVAL:
        await sync_positions(client)
        state["last_position_sync"] = now

    # ── Staggered indicator refresh: one symbol per tick ─────────
    if now - _last_indicator_tick >= INDICATOR_INTERVAL:
        sym = SYMBOLS[_indicator_index % len(SYMBOLS)]
        await update_one_symbol(client, sym)
        _indicator_index    += 1
        _last_indicator_tick = now

    # ── Sentiment: one stale symbol per tick (in executor) ───────
    stale_sent = min(SYMBOLS, key=lambda s: state["last_sentiment"].get(s, 0.0))
    if now - state["last_sentiment"].get(stale_sent, 0.0) >= SENTIMENT_INTERVAL:
        market_data[stale_sent]["sentiment"] = await get_AI_sentiment(stale_sent)
        state["last_sentiment"][stale_sent]  = now

    # ── Balance (cached) ─────────────────────────────────────────
    balance = await get_usdt_balance(client)

    # ── Trade logic (also handles symbol reshuffle) ───────────────
    await run_trading_logic(client, balance)

    # ── Deep AI analysis: one stale symbol per tick ──────────────
    stale_ai = min(SYMBOLS, key=lambda s: analysis_time.get(s, 0.0))
    if now - analysis_time.get(stale_ai, 0.0) >= ANALYSIS_INTERVAL:
        analyses[stale_ai]      = await get_deep_analysis(stale_ai, market_data[stale_ai]["price"])
        analysis_time[stale_ai] = now

    # ── Telegram: signal-change alerts + periodic status / log push ──
    if TELEGRAM_ENABLED:
        notify_signal_changes()
        if now - state.get("last_tg_status", 0.0) >= STATUS_INTERVAL:
            tg_schedule(tg_send_message(_tg_status_text() + "\n\n" + _tg_positions_text()))
            state["last_tg_status"] = now
        if now - state.get("last_tg_log", 0.0) >= LOG_SEND_INTERVAL:
            tg_schedule(tg_send_document(LOG_TXT_FILE, "📄 periodic bot.log"))
            state["last_tg_log"] = now

    # ── Render: dashboard (TTY) or plain status lines (headless/Docker) ──
    if HEADLESS:
        if now - state.get("last_plain_print", 0.0) >= HEADLESS_STATUS_INTERVAL:
            print_plain_status(balance)
            state["last_plain_print"] = now
    else:
        live.update(build_layout(balance))


async def ws_all_tickers(client: AsyncClient, live: Live, state: dict):
    """
    Subscribe to ALL active symbols' mini-ticker stream so every symbol gets a
    live price. When a reshuffle changes the symbol set, _ws_needs_restart is
    set and we re-subscribe in place (no full client teardown). A genuine socket
    drop bubbles up to main() for a full reconnect.
    """
    global _ws_needs_restart
    bm = BinanceSocketManager(client)

    while True:
        streams = [f"{s.lower()}@miniTicker" for s in SYMBOLS]
        try:
            async with bm.multiplex_socket(streams) as ms:
                _ws_needs_restart = False
                while True:
                    try:
                        res  = await asyncio.wait_for(ms.recv(), timeout=5)
                        data = res.get("data", {})
                        sym  = data.get("s", "")
                        if sym in market_data and "c" in data:
                            market_data[sym]["price"]    = float(data["c"])
                            market_data[sym]["price_ts"] = time.time()
                    except asyncio.TimeoutError:
                        pass
                    except (asyncio.CancelledError, KeyboardInterrupt):
                        raise

                    await trading_loop(client, live, state)

                    # Reshuffle happened → break inner loop and re-subscribe
                    if _ws_needs_restart:
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
            return   # let main() do a full reconnect

        # If we got here via _ws_needs_restart, loop and re-subscribe with new SYMBOLS


async def main():
    global _session_start_balance, _indicator_index, _last_indicator_tick

    _ensure_log()
    client = await AsyncClient.create(
        api_key=os.getenv("BINANCE_API_KEY"),
        api_secret=os.getenv("BINANCE_API_SECRET"),
        testnet=USE_TESTNET,
    )

    console.print(Panel.fit(
        f"[bold green]🚀 AI Futures Trader v4  |  "
        f"Leverage: {LEVERAGE}x  |  "
        f"Risk: {RISK_PERCENT*100:.1f}% / trade  |  "
        f"Active symbols: {len(SYMBOLS)}/{len(_SYMBOL_POOL)}  |  "
        f"Max L:{MAX_OPEN_LONGS} S:{MAX_OPEN_SHORTS}  |  "
        f"Reshuffle: {SYMBOL_RESHUFFLE_HOURS}h[/bold green]",
        style="cyan",
    ))

    # ── Validate Binance connection & permissions BEFORE trading ──────────────
    # Catches the -2015 "Invalid API-key, IP, or permissions" error at startup
    # instead of spamming it every few seconds in the trading loop.
    if not await verify_binance_access(client):
        try:
            await client.close_connection()
        except Exception:
            pass
        _executor.shutdown(wait=False)
        return

    # ── Prune delisted / untradable symbols (self-healing) ───────────────────
    await filter_tradable_symbols(client)
    console.print(f"[dim]🎲 Active symbols: {', '.join(SYMBOLS)}[/dim]")

    # Pre-warm symbol info cache for all symbols in one call
    console.print("[dim]Pre-loading symbol info…[/dim]")
    await get_symbol_info(client, SYMBOLS[0])   # this fills cache for ALL symbols

    await setup_leverage(client)
    await sync_positions(client)

    # Boot-strap first symbol immediately so display isn't blank
    console.print("[dim]Loading initial indicators (first symbol)…[/dim]")
    await update_one_symbol(client, SYMBOLS[0])
    _indicator_index    = 1
    _last_indicator_tick = time.time()

    _session_start_balance = await get_usdt_balance(client)
    console.print(f"[cyan]💰 Session start balance: ${_session_start_balance:,.2f} USDT[/cyan]")

    # ── Telegram startup ──────────────────────────────────────────
    if TELEGRAM_ENABLED:
        console.print("[green]📨 Telegram notifications ENABLED[/green]")
        notify(
            f"🚀 AI Futures Trader v4 started\n"
            f"Balance: ${_session_start_balance:,.2f} USDT | Leverage {LEVERAGE}x\n"
            f"Symbols ({len(SYMBOLS)}): {', '.join(SYMBOLS)}\n"
            f"Send /help for commands.", "INFO"
        )
    else:
        console.print("[yellow]📭 Telegram disabled (set TELEGRAM_BOT_TOKEN + "
                      "TELEGRAM_CHAT_ID in .env)[/yellow]")

    # Stagger initial AI analysis so they don't all fire on tick 1
    for i, sym in enumerate(SYMBOLS):
        analysis_time[sym] = time.time() - (i * (ANALYSIS_INTERVAL / max(len(SYMBOLS), 1)))

    state = {
        "last_position_sync": 0.0,
        # sentiment times stored as plain dict; .get(sym, 0.0) used everywhere
        "last_sentiment": {sym: float(i * SENTIMENT_INTERVAL / max(len(SYMBOLS), 1))
                           for i, sym in enumerate(SYMBOLS)},
        "last_tg_status": time.time(),   # first summary after STATUS_INTERVAL
        "last_tg_log":    time.time(),   # first log push after LOG_SEND_INTERVAL
        "last_plain_print": 0.0,         # headless status print throttle
    }

    # Launch the Telegram command poller as a background task (read-only)
    poller_task = asyncio.create_task(telegram_command_poller()) if TELEGRAM_ENABLED else None

    if HEADLESS:
        console.print("[cyan]🖥  Headless mode (no TTY) — printing plain status "
                      "lines. Dashboard disabled. Use `tail -f data/bot.log` too.[/cyan]")

    try:
        _live_ctx = _NoopLive() if HEADLESS else Live(
            console=console, refresh_per_second=2, screen=True
        )
        with _live_ctx as live:
            while True:
                try:
                    await ws_all_tickers(client, live, state)
                except (asyncio.CancelledError, KeyboardInterrupt):
                    raise
                except Exception as e:
                    console.print(f"[red]Loop error: {e}[/red]")

                console.print(f"[yellow]🔄 Reconnecting in {RECONNECT_DELAY}s…[/yellow]")
                for _ in range(RECONNECT_DELAY):
                    try:
                        balance = await get_usdt_balance(client)
                        live.update(build_layout(balance))
                    except Exception:
                        pass
                    try:
                        await asyncio.sleep(1)
                    except (asyncio.CancelledError, KeyboardInterrupt):
                        raise

                try:
                    await client.close_connection()
                except Exception:
                    pass
                try:
                    client = await AsyncClient.create(
                        api_key=os.getenv("BINANCE_API_KEY"),
                        api_secret=os.getenv("BINANCE_API_SECRET"),
                        testnet=USE_TESTNET,
                    )
                    console.print("[green]✅ Reconnected.[/green]")
                except Exception as e:
                    console.print(f"[red]Reconnect failed: {e}[/red]")

    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[bold red]Bot stopped.[/bold red]")
        final = await get_usdt_balance(client)
        col   = "green" if _daily_realized_pnl >= 0 else "red"
        console.print(
            f"[cyan]Session PnL: [{col}]${_daily_realized_pnl:+.2f}[/] | "
            f"Final balance: ${final:,.2f} USDT[/cyan]"
        )
        # Final Telegram report + bot.log file
        if TELEGRAM_ENABLED:
            write_log(f"Bot stopped. Session PnL ${_daily_realized_pnl:+.2f} | "
                      f"Final balance ${final:,.2f}", "INFO")
            await tg_send_message(
                f"🛑 Bot stopped\n"
                f"Session PnL: ${_daily_realized_pnl:+.2f}\n"
                f"Final balance: ${final:,.2f} USDT"
            )
            await tg_send_document(LOG_TXT_FILE, "📄 final bot.log")
    finally:
        if poller_task:
            poller_task.cancel()
        try:
            await client.close_connection()
        except Exception:
            pass
        _executor.shutdown(wait=False)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass