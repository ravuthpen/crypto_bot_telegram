import os
import random
import time
import asyncio

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

import pandas as pd
from binance import AsyncClient
from binance.enums import (
    ORDER_TYPE_MARKET,
    FUTURE_ORDER_TYPE_STOP_MARKET,
)
from rich.console import Console
from rich.panel import Panel

from config import BotConfig
from round_step import round_step

console = Console()

# Exchange-info changes rarely (new listings / delistings). Cache the raw
# payload so symbol_info() and filter_tradable() don't each download it.
_EXCHANGE_INFO_TTL = 3600.0  # seconds

# Used when a symbol's filters can't be read — conservative, valid-everywhere.
_DEFAULT_FILTERS = {"step_size": 0.001, "tick_size": 0.01,
                    "min_qty": 0.001, "min_notional": 5.0}


class Exchange:
    """Async wrapper over python-binance: connection, balance/margin caching,
    symbol filters, leverage cache, order placement, and kline fetch."""

    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.client: AsyncClient | None = None
        self._sym_info_cache: dict[str, dict] = {}
        self._leverage_done: set[str] = set()
        self.cached_balance: float = 0.0
        self._cached_balance_ts: float = 0.0
        self._exchange_info: dict | None = None
        self._exchange_info_ts: float = 0.0

    async def connect(self):
        self.client = await asyncio.wait_for(
            AsyncClient.create(
                api_key=os.getenv("BINANCE_API_KEY"),
                api_secret=os.getenv("BINANCE_API_SECRET"),
                testnet=self.cfg.use_testnet,
            ),
            timeout=15,
        )

    async def close(self):
        if self.client:
            try:
                await self.client.close_connection()
            except Exception:
                pass

    def invalidate_balance_cache(self):
        self._cached_balance_ts = 0.0

    # ── verification ───────────────────────────────────────────────────────
    async def verify_access(self) -> bool:
        console.print("[dim]Verifying Binance API access…[/dim]")
        try:
            bals = await self.client.futures_account_balance()
            usdt = next((float(b["balance"]) for b in bals if b["asset"] == "USDT"), 0.0)
            console.print(f"[green]✅ Binance API OK — Futures USDT balance: ${usdt:,.2f}[/green]")
            return True
        except Exception as e:
            msg = str(e)
            console.print(Panel.fit(
                f"[bold red]❌ Binance API check failed[/bold red]\n\n{msg}\n\n"
                f"[yellow]If this is -2015 'Invalid API-key, IP, or permissions':[/yellow]\n"
                f"  1. Key/secret correct and matches USE_TESTNET={self.cfg.use_testnet}?\n"
                f"  2. Is Futures permission enabled on the API key?\n"
                f"  3. Is this server's IP in the key's allowed-IP list?\n"
                f"  4. On a VPS: check egress IP with  curl -s https://api.ipify.org\n",
                style="red",
            ))
            return False

    # ── exchange info (shared, cached) ───────────────────────────────────────
    async def exchange_info(self, force: bool = False) -> dict:
        """Return futures_exchange_info, cached for _EXCHANGE_INFO_TTL seconds.
        Pass force=True to bypass the cache (used for delisting self-heal)."""
        now = time.time()
        if (not force and self._exchange_info is not None
                and now - self._exchange_info_ts < _EXCHANGE_INFO_TTL):
            return self._exchange_info
        info = await self.client.futures_exchange_info()
        self._exchange_info = info
        self._exchange_info_ts = now
        return info

    # ── symbol metadata ─────────────────────────────────────────────────────
    async def filter_tradable(self, symbol_pool: list[str], active_symbols: list[str],
                              active_count: int) -> tuple[list[str], list[str], list[str]]:
        """Return (live_pool, new_active_symbols, dead_symbols)."""
        try:
            info = await self.exchange_info(force=True)   # fresh: catch delistings
        except Exception as e:
            console.print(f"[yellow]⚠ Could not verify symbols (keeping as-is): {e}[/yellow]")
            return symbol_pool, active_symbols, []

        tradable = {
            s["symbol"] for s in info.get("symbols", [])
            if s.get("status") == "TRADING"
            and s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
        }
        dead = [s for s in symbol_pool if s not in tradable]
        live_pool = [s for s in symbol_pool if s in tradable]
        if not live_pool:
            console.print("[red]❌ No tradable symbols left in pool! Keeping original.[/red]")
            return symbol_pool, active_symbols, dead

        kept = [s for s in active_symbols if s in tradable]
        need = max(0, min(active_count, len(live_pool)) - len(kept))
        extra = [s for s in live_pool if s not in kept]
        random.shuffle(extra)
        new_active = kept + extra[:need]
        return live_pool, new_active, dead

    @staticmethod
    def _parse_filters(s: dict) -> dict | None:
        """Extract LOT_SIZE / PRICE_FILTER / MIN_NOTIONAL from one symbol entry.
        Returns None if the mandatory filters are absent (so the caller can skip
        this symbol rather than aborting the whole cache build)."""
        f = {x["filterType"]: x for x in s.get("filters", [])}
        lot, price = f.get("LOT_SIZE"), f.get("PRICE_FILTER")
        if not lot or not price:
            return None
        return {
            "step_size":    float(lot["stepSize"]),
            "tick_size":    float(price["tickSize"]),
            "min_qty":      float(lot["minQty"]),
            "min_notional": float(f.get("MIN_NOTIONAL", {}).get("notional", 5)),
        }

    async def symbol_info(self, symbol: str) -> dict:
        """LOT_SIZE / PRICE_FILTER for a symbol; first call caches ALL symbols.
        One malformed symbol no longer poisons the cache for the rest."""
        if symbol in self._sym_info_cache:
            return self._sym_info_cache[symbol]
        try:
            info = await self.exchange_info()
            for s in info.get("symbols", []):
                parsed = self._parse_filters(s)
                if parsed is not None:
                    self._sym_info_cache[s["symbol"]] = parsed
            if symbol in self._sym_info_cache:
                return self._sym_info_cache[symbol]
        except Exception as e:
            console.print(f"[yellow]⚠ symbol_info {symbol}: {e}[/yellow]")
        # Negative-cache the fallback so we don't re-download on every miss.
        console.print(f"[yellow]⚠ {symbol}: filters unavailable, using defaults[/yellow]")
        self._sym_info_cache[symbol] = dict(_DEFAULT_FILTERS)
        return self._sym_info_cache[symbol]

    async def ensure_leverage(self, symbol: str):
        if symbol in self._leverage_done:
            return
        try:
            await self.client.futures_change_leverage(symbol=symbol, leverage=self.cfg.leverage)
            self._leverage_done.add(symbol)
        except Exception as e:
            # Don't mark done on failure: position sizing assumes cfg.leverage is
            # actually set, so a silent failure here can over-size and trigger -2019.
            console.print(f"[yellow]⚠ leverage {symbol}: {e}[/yellow]")

    # ── balances ─────────────────────────────────────────────────────────────
    async def usdt_balance(self) -> float:
        if time.time() - self._cached_balance_ts < self.cfg.balance_refresh_interval:
            return self.cached_balance
        try:
            bals = await self.client.futures_account_balance()
            for b in bals:
                if b["asset"] == "USDT":
                    self.cached_balance = float(b["balance"])
                    break
            # Stamp the cache even if USDT wasn't present, so a balance-less
            # account doesn't hammer the API every call.
            self._cached_balance_ts = time.time()
        except Exception as e:
            console.print(f"[yellow]⚠ balance: {e}[/yellow]")
        return self.cached_balance

    async def available_margin(self) -> float:
        """Free USDT margin for NEW positions (what -2019 actually checks)."""
        try:
            for b in await self.client.futures_account_balance():
                if b["asset"] == "USDT":
                    return float(b.get("availableBalance", b.get("balance", 0.0)))
        except Exception as e:
            console.print(f"[yellow]⚠ avail margin: {e}[/yellow]")
        return 0.0

    async def realized_pnl(self, symbol: str, since_ms: int) -> float:
        """Ground-truth NET realized PnL for `symbol` since `since_ms` (ms epoch),
        straight from the exchange income ledger: realized PnL + fees + funding.

        Use this to reconcile self.daily_realized_pnl instead of trusting the
        scaled-unrealized estimate. Note the ledger can lag a market fill by a
        moment, so query on the next sync rather than the instant after closing."""
        try:
            rows = await self.client.futures_income(
                symbol=symbol, startTime=int(since_ms), limit=1000)
            return sum(
                float(r["income"]) for r in rows
                if r.get("incomeType") in ("REALIZED_PNL", "COMMISSION", "FUNDING_FEE")
            )
        except Exception as e:
            console.print(f"[yellow]⚠ realized_pnl {symbol}: {e}[/yellow]")
            return 0.0

    # ── orders ────────────────────────────────────────────────────────────────
    async def positions_info(self) -> list[dict]:
        return await self.client.futures_position_information()

    async def cancel_open_orders(self, symbol: str):
        try:
            await self.client.futures_cancel_all_open_orders(symbol=symbol)
        except Exception:
            pass

    async def market_order(self, symbol: str, side: str, qty: float,
                           reduce_only: bool = False) -> bool:
        """Market order. Pass reduce_only=True for CLOSES so the order can only
        shrink an existing position — never flip it open in the other direction
        if the position is already gone or smaller than we think."""
        try:
            params = dict(symbol=symbol, side=side.upper(),
                          type=ORDER_TYPE_MARKET, quantity=qty)
            if reduce_only:
                params["reduceOnly"] = "true"
            await self.client.futures_create_order(**params)
            col = "green" if side.upper() == "BUY" else "red"
            tag = " (reduceOnly)" if reduce_only else ""
            console.print(f"[bold {col}]✅ {side.upper()} {symbol} qty={qty}{tag}[/]")
            return True
        except Exception as e:
            console.print(f"[red]❌ order failed {symbol} {side}: {e}[/red]")
            return False

    async def place_stop_loss(self, symbol: str, pos_side: str, sl_price: float) -> int | None:
        """Hard catastrophic SL on the exchange (survives a bot crash).
        TP1/TP2/trail stay in software to avoid a double-close race."""
        close_side = "SELL" if pos_side == "LONG" else "BUY"
        info = await self.symbol_info(symbol)
        sl_p = round_step(sl_price, info["tick_size"])
        try:
            r = await self.client.futures_create_order(
                symbol=symbol, side=close_side,
                type=FUTURE_ORDER_TYPE_STOP_MARKET,
                stopPrice=sl_p, closePosition=True,
                timeInForce="GTE_GTC", workingType="MARK_PRICE",
            )
            return r.get("orderId")
        except Exception as e:
            console.print(f"[yellow]⚠ SL order {symbol}: {e}[/yellow]")
            return None

    async def replace_stop_loss(self, symbol: str, pos_side: str, new_sl: float) -> int | None:
        await self.cancel_open_orders(symbol)
        return await self.place_stop_loss(symbol, pos_side, new_sl)

    # ── klines ────────────────────────────────────────────────────────────────
    async def fetch_df(self, symbol: str, interval: str, limit: int = 250) -> pd.DataFrame | None:
        try:
            klines = await self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            if not klines:
                return None
            return pd.DataFrame(klines, columns=[
                "open_time", "Open", "High", "Low", "Close", "Volume",
                "close_time", "qav", "n_trades", "taker_base", "taker_quote", "ignore",
            ]).astype(float)
        except Exception as e:
            console.print(f"[yellow]⚠ fetch_df {symbol} {interval}: {e}[/yellow]")
            return None