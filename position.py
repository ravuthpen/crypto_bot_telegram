from dataclasses import dataclass, field


@dataclass
class Position:
    side: str | None = None              # "LONG" | "SHORT" | None
    qty: float = 0.0
    entry: float = 0.0
    sl_price: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    trail_active: bool = False
    trail_best: float = 0.0
    trail_stop: float = 0.0
    pnl_pct: float = 0.0
    pnl_usdt: float = 0.0
    tp1_hit: bool = False
    sl_order_id: int | None = None
    tp1_order_id: int | None = None
    open_time: float = 0.0

    # NEW: Track runner mode state
    runner_active: bool = False          # True if TP2 partial hit, rest trailing
    runner_qty: float = 0.0              # Qty still running with trail

    # NEW: Track breakeven state after TP1
    be_hit: bool = False                 # True if SL moved to breakeven+

    # NEW: Track which AI model signaled this trade
    ai_model: str = ""                   # "chatgpt", "grok", "consensus", or ""

    # NEW: Original signal for performance tracking
    ai_signal: dict = field(default_factory=dict)

    # NEW: Track order IDs for TP orders (if exchange supports)
    tp1_order_id: int | None = None
    tp2_order_id: int | None = None

    # NEW: Partial close tracking
    partials_closed: list[dict] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.side is not None and self.qty > 0

    @property
    def remaining_risk(self) -> float:
        """Calculate remaining risk in USDT."""
        if not self.is_open or self.entry <= 0:
            return 0.0
        sl_dist = abs(self.entry - self.sl_price)
        return sl_dist * self.qty

    @property
    def current_rr(self) -> float:
        """Current risk/reward based on active targets."""
        if not self.is_open or self.sl_price == 0:
            return 0.0
        risk = abs(self.entry - self.sl_price)
        if risk == 0:
            return 0.0
        # If runner active, no fixed TP — use trail
        if self.runner_active:
            return 0.0  # Undefined, managed by trail
        target = self.tp2_price if self.tp1_hit else self.tp1_price
        reward = abs(target - self.entry)
        return reward / risk

    def reset(self):
        """Reset position to closed state."""
        self.side = None
        self.qty = 0.0
        self.entry = 0.0
        self.sl_price = 0.0
        self.tp1_price = 0.0
        self.tp2_price = 0.0
        self.trail_active = False
        self.trail_best = 0.0
        self.trail_stop = 0.0
        self.pnl_pct = 0.0
        self.pnl_usdt = 0.0
        self.tp1_hit = False
        self.be_hit = False
        self.runner_active = False
        self.runner_qty = 0.0
        self.sl_order_id = None
        self.tp1_order_id = None
        self.tp2_order_id = None
        self.ai_model = ""
        self.ai_signal = {}
        self.partials_closed = []

    def record_partial(self, qty: float, price: float, reason: str):
        """Record a partial close for tracking."""
        self.partials_closed.append({
            "qty": qty,
            "price": price,
            "reason": reason,
            "time": __import__('time').time()
        })