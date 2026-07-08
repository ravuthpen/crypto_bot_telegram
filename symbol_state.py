import time
from dataclasses import dataclass, field

from position import Position


@dataclass
class SymbolState:
    symbol: str
    price: float = 0.0
    price_ts: float = 0.0
    rsi_15: float = 50.0
    rsi_1h: float = 50.0
    sma20: float = 0.0
    ema50: float = 0.0
    ema200: float = 0.0
    macd: float = 0.0
    macd_sig: float = 0.0
    atr: float = 0.0
    adx: float = 0.0
    bb_squeeze: bool = False
    volume_ok: bool = False
    signal: str = "HOLD"
    hold_reason: str = ""
    sentiment: str = "NEUTRAL"
    ind_ts: float = 0.0
    position: Position = field(default_factory=Position)
    
    # FIX: analysis is now dict (from AIAnalyst), not string
    analysis: dict = field(default_factory=dict)
    analysis_ts: float = 0.0
    cooldown_until: float = 0.0

    # NEW: Track last AI recommendation that led to a trade
    last_ai_entry: dict = field(default_factory=dict)
    
    # NEW: Performance tracking for this symbol
    trade_count: int = 0
    win_count: int = 0
    total_pnl: float = 0.0

    def reset_position(self):
        # FIX: Use reset() instead of creating new object (preserves reference)
        self.position.reset()

    def price_is_fresh(self, threshold: float) -> bool:
        return time.time() - self.price_ts < threshold

    # NEW: Record trade result for symbol-level performance
    def record_trade(self, pnl_usdt: float):
        self.trade_count += 1
        self.total_pnl += pnl_usdt
        if pnl_usdt > 0:
            self.win_count += 1

    @property
    def symbol_win_rate(self) -> float:
        return (self.win_count / self.trade_count * 100) if self.trade_count > 0 else 0.0

    # NEW: Check if AI analysis is fresh and actionable
    def ai_is_fresh(self, max_age: float = 180) -> bool:
        return (
            bool(self.analysis) 
            and time.time() - self.analysis_ts < max_age
            and self.analysis.get("action") in ("LONG", "SHORT")
        )

    # NEW: Get formatted AI action for display
    def ai_action_str(self) -> str:
        if not self.analysis:
            return "WAIT"
        return self.analysis.get("action", "WAIT")

    # NEW: Get AI confidence safely
    def ai_confidence(self) -> int:
        if not self.analysis:
            return 0
        return int(self.analysis.get("confidence", 0))