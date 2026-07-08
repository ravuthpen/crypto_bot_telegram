import os
import sys
from pathlib import Path

class BotConfig:
    # ── Master symbol pool — bot picks randomly from here ──────────────────
    symbol_pool: tuple[str, ...] = (
        "AVAXUSDT", "BTCUSDT",  "ETHUSDT",  "BNBUSDT",  "SOLUSDT",
        "XRPUSDT",  "DOGEUSDT", "ADAUSDT",  "LINKUSDT", "LTCUSDT",
        "TRXUSDT",  "DOTUSDT",  "POLUSDT",  "ATOMUSDT", "NEARUSDT",
        "ARBUSDT",  "OPUSDT",   "APTUSDT",  "SUIUSDT",  "1000PEPEUSDT",
        "BNXUSDT",  "INJUSDT",  "TIAUSDT",  "SEIUSDT",  "WIFUSDT",
        "JUPUSDT",  "PYTHUSDT", "STRKUSDT", "FETUSDT",  "RENDERUSDT",
    )
    
    # FIXED: Match active count to max positions + buffer for selection
    # 3 long + 3 short = 6 max, +4 for rotation = 10 active symbols max
    active_symbol_count: int = 10          # Was 20 — too many for 6 max positions
    symbol_reshuffle_hours: float = 4.0

    use_testnet: bool = False
    leverage: int = 5                      # Was 3 — 5x is standard for futures risk management
    risk_percent: float = 0.015            # Was 0.01 — slightly higher for better returns with wider TPs
    margin_safety: float = 0.90            # Was 0.95 — leave more headroom for volatility

    # ── Take Profit / Stop Loss ────────────────────────────────────────────
    atr_sl_mult: float = 1.5
    atr_tp1_mult: float = 3.0              # 2:1 R:R minimum (was 2.0, too tight)
    atr_tp2_mult: float = 6.0              # 4:1 R:R (was 4.0)
    
    # ── Trailing Stop ──────────────────────────────────────────────────────
    trail_activation_mult: float = 0.5     # Activate earlier (was probably higher)
    trail_offset_mult: float = 0.6         # Tighter trail to lock profit
    
    # NEW: Runner mode — after TP2, close partial and let rest trail
    runner_fraction: float = 0.50          # Close 50% at TP2, trail remaining 50%
    runner_trail_mult: float = 0.4         # Even tighter trail for runner (0.4x ATR)
    
    # NEW: Breakeven buffer after TP1 partial
    breakeven_buffer_pct: float = 0.003    # 0.3% above entry to avoid stop hunting

    # ── Position Limits ────────────────────────────────────────────────────
    max_open_longs: int = 3
    max_open_shorts: int = 3
    max_correlated_positions: int = 3      # NEW: Prevent same-direction stacking

    # ── Trading Parameters ─────────────────────────────────────────────────
    min_usdt_to_trade: float = 20.0
    adx_min: float = 20.0
    squeeze_lookback: int = 20
    bb_std: float = 2.0
    partial_close_fraction: float = 0.35   # Was 0.5 — close less at TP1, let more run
    
    # NEW: Minimum R:R required to take a trade (tech + AI must meet this)
    min_rr_ratio: float = 2.0
    
    cooldown_after_loss: int = 600         # Was 300 — 10 min cooldown to avoid revenge trading
    daily_loss_limit_pct: float = 0.05
    stale_price_threshold: int = 60

    # ── Intervals (seconds) ────────────────────────────────────────────────
    analysis_interval: int = 90
    position_sync_interval: int = 30       # Was 8 — avoid Binance rate limits (1200/min max)
    balance_refresh_interval: int = 30
    indicator_interval: int = 30
    sentiment_interval: int = 300
    reconnect_delay: int = 5

    # ── Timeframes ─────────────────────────────────────────────────────────
    tf_primary: str = "15m"
    tf_confirm: str = "1h"
    
    # ── Logging ────────────────────────────────────────────────────────────
    log_file: Path = Path("trades.csv")
    log_txt_file: Path = Path("bot.log")

    # ── Safety ─────────────────────────────────────────────────────────────
    position_adopt_grace: int = 10         # sec: open→sync race guard

    # ── Exchange Fees ──────────────────────────────────────────────────────
    # NEW: Binance futures taker fee (0.04% for regular, 0.036% for BNB discount)
    taker_fee_rate: float = 0.0004
    maker_fee_rate: float = 0.0002
    use_bnb_discount: bool = False         # Set True if paying fees with BNB
    
    # NEW: Slippage allowance for market orders
    max_slippage_pct: float = 0.001        # 0.1% max slippage before warning

    # ── Telegram ───────────────────────────────────────────────────────────
    tg_status_interval: int = 1800
    tg_log_send_interval: int = 3600
    tg_signal_debounce: int = 60
    headless_status_interval: int = 30

    # ── AI Models (FIXED: separate env vars) ──────────────────────────────
    @property
    def chat_model(self) -> str:
        """OpenAI model for ChatGPT analysis."""
        return os.getenv("CHAT_MODEL")      # FIXED: was MODEL, now CHAT_MODEL

    @property
    def grok_model(self) -> str:
        """xAI model for Grok analysis."""
        return os.getenv("GROK_MODEL")       # FIXED: was MODEL, now GROK_MODEL

    @property
    def ai_model(self) -> str:
        """Fallback model (used by legacy code)."""
        return self.grok_model

    # ── AI Thresholds (NEW) ───────────────────────────────────────────────
    ai_min_confidence: int = 70              # Minimum AI confidence to consider signal
    ai_strong_confidence: int = 85           # High confidence for override decisions
    ai_disagree_block: bool = True           # Block trade if AIs disagree on bias
    
    # NEW: AI performance tracking
    ai_min_history: int = 10                 # Min trades before adapting thresholds
    ai_win_rate_target: float = 0.55         # Target win rate for AI signals

    # ── Telegram (unchanged) ──────────────────────────────────────────────
    @property
    def telegram_token(self) -> str:
        return os.getenv("TELEGRAM_BOT_TOKEN", "")

    @property
    def telegram_chat_id(self) -> str:
        return os.getenv("TELEGRAM_CHAT_ID", "")

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_token and self.telegram_chat_id)

    # ── Headless Detection (unchanged) ────────────────────────────────────
    def resolve_headless(self) -> bool:
        env = os.getenv("DASHBOARD", "")
        if env in ("0", "false", "False"):
            return True
        if env in ("1", "true", "True"):
            return False
        return not sys.stdout.isatty()   # auto: no TTY (Docker) → headless