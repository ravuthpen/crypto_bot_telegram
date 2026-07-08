import csv
import os
import threading
from pathlib import Path


class TradeLogger:
    """Append-only CSV trade log.

    Synchronous on purpose: a single log() call runs to completion without
    awaiting, so it's safe to call from inside an async trading loop without
    records interleaving, and a threading.Lock guards the rare off-thread call.
    """

    FIELDS = [
        "date", "time", "symbol", "action", "side", "entry", "exit",
        "qty", "sl", "tp1", "tp2", "pnl_pct", "pnl_usdt", "reason",
    ]

    def __init__(self, path: Path):
        self.path = Path(path)              # also accepts a str
        self._lock = threading.Lock()
        self._ensure()

    def _ensure(self):
        """Create the parent dir + header row if the file is missing or empty."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()
                f.flush()
                os.fsync(f.fileno())

    def log(self, row: dict):
        safe = {k: _clean(row.get(k)) for k in self.FIELDS}
        with self._lock:
            self._ensure()                  # re-create header if file was rotated/deleted
            with self.path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writerow(safe)
                f.flush()
                os.fsync(f.fileno())         # durability: don't lose a trade on a crash


def _clean(v):
    """Keep every value on one CSV line and free of None/NaN surprises."""
    if v is None:
        return ""
    if isinstance(v, float) and v != v:     # NaN
        return ""
    return str(v).replace("\r", " ").replace("\n", " ")