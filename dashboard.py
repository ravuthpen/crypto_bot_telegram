import os
import time
from dotenv import load_dotenv
from rich.console import Console
from rich.console import Group as RichGroup
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.columns import Columns
from datetime import datetime
from symbol_state import SymbolState
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading_bot import TradingBot

load_dotenv()
console = Console()


# ── safe numeric helpers ───────────────────────────────────────────────────
def _to_float(v, default: float = 0.0) -> float:
    """Coerce ints, floats, or formatted strings ('$1,234.56', '+2.3%') to float.

    Trade-log values are sometimes stored already formatted (with $ , % +). A bare
    float() on those raises and — in panels without an error boundary — takes down
    the whole render. This never raises; it returns `default` on failure.
    """
    if isinstance(v, (int, float)):
        return float(v)
    if v is None:
        return default
    try:
        return float(str(v).replace("$", "").replace(",", "").replace("%", "").replace("+", "").strip())
    except (ValueError, AttributeError):
        return default


def _trade_pnl_value(t: dict):
    """Return a trade's realized PnL as a number, or None if it's not a resolved trade.

    Resolved == has a usable pnl/pnl_usdt field. OPEN events return None so they are
    excluded from win-rate denominators. Tolerates numeric or formatted-string PnL,
    under either the `pnl_usdt` or `pnl` key.
    """
    for key in ("pnl_usdt", "pnl"):
        raw = t.get(key)
        if raw in (None, ""):
            continue
        try:
            return float(str(raw).replace("$", "").replace(",", "").replace("%", "").replace("+", "").strip())
        except (ValueError, AttributeError):
            continue
    return None


class Dashboard:
    def __init__(self, bot: "TradingBot"):
        self.bot = bot

    @staticmethod
    def _term_size() -> tuple[int, int]:
        try:
            s = os.get_terminal_size()
            return max(s.columns, 80), max(s.lines, 24)  # FIX: minimum bounds
        except OSError:
            return 120, 30  # FIX: more reasonable fallback

    # ── safe analysis accessor ─────────────────────────────────────────────
    def _get_ai_field(self, s: SymbolState, key: str, default: str = "—") -> str:
        """Safely extract field from analysis regardless of format."""
        raw = s.analysis
        
        # NEW: Handle dict format (from fixed AIAnalyst)
        if isinstance(raw, dict):
            return str(raw.get(key, default))
        
        # OLD: Fallback to text parsing if still string
        try:
            from extract_field import extract_field
            return extract_field(raw, key) or default
        except Exception:
            return default

    def _get_ai_block(self, s: SymbolState, key: str) -> list[str]:
        """Safely extract block from analysis."""
        raw = s.analysis
        if isinstance(raw, dict):
            # Dict format doesn't have blocks, return empty
            return []
        try:
            from extract_block import extract_block
            return extract_block(raw, key) or []
        except Exception:
            return []

    # ── market regime display ──────────────────────────────────────────────
    def _regime_text(self, s: SymbolState) -> tuple[str, str]:
        """Return regime label and color."""
        adx = s.adx
        squeeze = s.bb_squeeze
        
        if adx >= 30 and squeeze:
            return "TREND+SQZ", "bold magenta"
        elif adx >= 25:
            return "TRENDING", "bold green"
        elif adx < 20 and squeeze:
            return "RNG+SQZ", "bold yellow"
        elif adx < 20:
            return "RANGING", "dim yellow"
        else:
            return "MIXED", "dim white"

    # ── per-symbol analysis card ───────────────────────────────────────────
    def _analysis_card(self, s: SymbolState) -> Panel:
        """Build analysis card with error boundary."""
        try:
            return self._analysis_card_inner(s)
        except Exception as e:
            # FIX: Error boundary - one bad card doesn't crash dashboard
            error_card = Table.grid(padding=(0, 1))
            error_card.add_column()
            error_card.add_row(f"[red]Error rendering {s.symbol}[/red]")
            error_card.add_row(f"[dim]{str(e)[:60]}[/dim]")
            return Panel(error_card, title=f"[bold]⚠️ {s.symbol}[/bold]", border_style="red")

    def _analysis_card_inner(self, s: SymbolState) -> Panel:
        pos = s.position
        cfg = self.bot.cfg

        # FIX: Use structured dict access with fallback
        bias = self._get_ai_field(s, "bias")
        confidence = self._get_ai_field(s, "confidence")
        action = self._get_ai_field(s, "action")
        entry_z = self._get_ai_field(s, "entry")
        sl = self._get_ai_field(s, "sl")
        tp1 = self._get_ai_field(s, "tp1")
        tp2 = self._get_ai_field(s, "tp2")
        reason = self._get_ai_field(s, "reason")
        
        # FIX: Handle numeric confidence
        try:
            conf_num = int(float(confidence)) if confidence != "—" else 0
        except ValueError:
            conf_num = 0

        bias_col = {"BULLISH": "green", "BEARISH": "red"}.get(
            str(bias).upper().split()[0] if bias else "", "yellow")
        action_col = {"LONG": "green", "SHORT": "red", "WAIT": "yellow"}.get(
            str(action).upper().split()[0] if action else "", "white")
        
        age_s = int(time.time() - s.analysis_ts)
        age_col = "green" if age_s < 120 else "yellow" if age_s < 300 else "red"
        ind_age = int(time.time() - s.ind_ts) if s.ind_ts > 0 else -1

        card = Table.grid(padding=(0, 1))
        card.add_column(style="dim", width=14, no_wrap=True)
        card.add_column(overflow="fold")

        def row(label, value, style=""):
            v = f"[{style}]{value}[/]" if style else str(value)
            card.add_row(f"[dim]{label}[/dim]", v)

        def sep(title=""):
            line = f"── {title} " + "─" * max(2, 20 - len(title)) if title else "─" * 22
            card.add_row("", f"[dim]{line}[/dim]")

        # ── Price & Market ─────────────────────────────────────────────────
        row("Price", f"${s.price:,.4f}", "bold white")
        
        # NEW: Show regime
        regime, regime_col = self._regime_text(s)
        row("Regime", regime, regime_col)
        
        ind_str = f"ind {ind_age}s ago" if ind_age >= 0 else "pending"
        row("Data age", ind_str, "green" if ind_age < 60 else "yellow" if ind_age < 120 else "red")
        
        # ── Indicators ─────────────────────────────────────────────────────
        row("RSI 15m/1h", f"{s.rsi_15:.1f} / {s.rsi_1h:.1f}")
        row("SMA20/EMA50", f"{s.sma20:,.2f} / {s.ema50:,.2f}")
        row("EMA200", f"{s.ema200:,.2f}")
        row("MACD/Sig", f"{s.macd:.3f} / {s.macd_sig:.3f}")
        row("ATR / ADX", f"{s.atr:.4f} / {s.adx:.1f}")
        row("BB Squeeze", "✓ YES" if s.bb_squeeze else "✗ NO", 
            "cyan" if s.bb_squeeze else "dim")
        row("Vol > Avg", "✓" if s.volume_ok else "✗", 
            "green" if s.volume_ok else "red")
        row("Sentiment", s.sentiment,
            "green" if s.sentiment == "BULLISH" else "red" if s.sentiment == "BEARISH" else "yellow")

        # ── AI Analysis ────────────────────────────────────────────────────
        sep("AI Analysis")
        row("Bias", f"{bias} ({conf_num}%)", bias_col)
        row("Action", action, action_col)
        
        # NEW: Show if AI is being used or not
        if not s.analysis or age_s > 600:
            row("Status", "⚠️ STALE / NO DATA", "red")
        elif conf_num < cfg.ai_min_confidence:
            row("Status", f"⚠️ Low confidence (<{cfg.ai_min_confidence})", "yellow")

        # ── Trade Levels ───────────────────────────────────────────────────
        sep("Trade Levels")
        if s.signal == "HOLD" and s.hold_reason:
            sig_disp, sig_col = f"HOLD (blocked: {s.hold_reason})", "yellow"
        else:
            sig_disp = s.signal
            sig_col = {"BUY": "green", "SHORT": "red",
                       "SELL_LONG": "red", "COVER_SHORT": "green"}.get(s.signal, "dim")
        row("Live Signal", sig_disp, sig_col)
        # FIX: use safe coercion so an AI level like "N/A" or "1,234.5" can't crash the card
        row("Entry Zone", f"${_to_float(entry_z):,.4f}" if entry_z != "—" else "—")
        row("Stop Loss", f"${_to_float(sl):,.4f}" if sl != "—" else "—", "red")
        row("TP1", f"${_to_float(tp1):,.4f}" if tp1 != "—" else "—", "green")
        row("TP2", f"${_to_float(tp2):,.4f}" if tp2 != "—" else "—", "green")

        # ── Open Position ──────────────────────────────────────────────────
        if pos.is_open:
            # FIX: Calculate real-time PnL instead of cached
            current_pnl_pct = ((s.price - pos.entry) / pos.entry * 100) if pos.side == "LONG" \
                else ((pos.entry - s.price) / pos.entry * 100) if pos.side == "SHORT" else 0
            current_pnl_usdt = current_pnl_pct / 100 * pos.qty * pos.entry
            
            pnl_col = "green" if current_pnl_pct >= 0 else "red"
            sep("Position")
            row("Side", pos.side, "cyan")
            row("Qty", f"{pos.qty:,.4f}")
            row("Entry", f"${pos.entry:,.4f}")
            row("SL/TP1", f"${pos.sl_price:,.4f} / ${pos.tp1_price:,.4f}")
            
            # FIX: Clearer trailing indicator
            if pos.trail_active:
                row("Trail", f"ACTIVE @ ${pos.trail_stop:,.4f}", "magenta")
            else:
                row("Trail", "inactive", "dim")
                
            row("TP1 Hit", "✓ YES" if pos.tp1_hit else "✗ NO", 
                "green" if pos.tp1_hit else "dim")
            row("PnL", f"{current_pnl_pct:+.2f}%  (${current_pnl_usdt:+.2f})", pnl_col)

        # ── Reasoning ──────────────────────────────────────────────────────
        if reason and reason != "—":
            sep("Reasoning")
            try:
                from wrap_text import wrap_text
                for wl in wrap_text(str(reason), 45):
                    card.add_row("", f"[italic dim]{wl}[/italic dim]")
            except Exception:
                card.add_row("", f"[italic dim]{str(reason)[:90]}[/italic dim]")

        card.add_row("", f"[{age_col}][dim]AI updated {age_s}s ago[/dim][/]")
        
        # FIX: Border color shows if AI is fresh
        border = "green" if age_s < 120 else "yellow" if age_s < 300 else "red"
        return Panel(card, title=f"[bold]🧠 {s.symbol}[/bold]", 
                     border_style=border, expand=True, padding=(0, 1))

    # ── summary table ────────────────────────────────────────────────────
    def _summary_table(self, balance: float, width: int) -> Table:
        bot, cfg = self.bot, self.bot.cfg
        sig_col = {"BUY": "bold green", "SHORT": "bold red", "HOLD": "dim white",
                   "SELL_LONG": "red", "COVER_SHORT": "green"}
        sent_col = {"BULLISH": "green", "BEARISH": "red", "NEUTRAL": "white"}  # FIX: NEUTRAL=white
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        daily = bot.daily_realized_pnl
        daily_s = f"[green]+${daily:.2f}[/]" if daily >= 0 else f"[red]-${abs(daily):.2f}[/]"
        halted = "  [bold red]⛔ HALTED[/bold red]" if bot.trading_halted else ""
        n_long, n_short = bot.count_open_longs(), bot.count_open_shorts()
        reshuffle_in = max(0, cfg.symbol_reshuffle_hours * 3600 - (time.time() - bot.last_reshuffle))
        reshuffle_m = int(reshuffle_in / 60)

        # FIX: win rate over RESOLVED trades only (exclude OPEN events) with robust
        # PnL parsing (works whether pnl is numeric, "+1.2", or stored under `pnl`).
        resolved_pnls = [v for v in (_trade_pnl_value(t) for t in bot.trade_log) if v is not None]
        total_trades = len(resolved_pnls)
        win_count = sum(1 for v in resolved_pnls if v > 0)
        win_rate = f"{win_count / total_trades * 100:.0f}%" if total_trades > 0 else "N/A"

        table = Table(
            title=(f"[bold]⚡ AI Futures Trader v5.1[/bold]  💰 ${balance:,.2f}  "
                   f"📈 Daily: {daily_s}  🕐 {ts}  📊 {cfg.leverage}x  "
                   f"[green]L:{n_long}/{cfg.max_open_longs}[/]  "
                   f"[red]S:{n_short}/{cfg.max_open_shorts}[/]  "
                   f"🏆 WR:{win_rate}  "
                   f"[dim]🔀 reshuffle {reshuffle_m}m[/dim]{halted}"),
            show_lines=True, expand=True, border_style="dim",
        )
        
        # FIX: Added AI confidence column
        base_cols = [("Symbol", 12, "center"), ("Price", 12, "right"),
                     ("RSI 15m", 7, "right"), ("ADX", 6, "right"),
                     ("Regime", 10, "center"), ("AI Conf", 7, "right"),
                     ("Signal", 15, "center"), ("Position", 8, "center"),
                     ("PnL %", 8, "right"), ("PnL USDT", 9, "right")]
        
        for col, mw, j in base_cols:
            table.add_column(col, justify=j, min_width=mw,
                             style="bold cyan" if col == "Symbol" else None)
                             
        if width >= 140:
            for col, mw in [("Sent", 9), ("Entry", 12), ("SL", 10), ("TP1", 10)]:
                table.add_column(col, justify="right" if col != "Sent" else "center", min_width=mw)
        if width >= 180:
            for col, mw in [("Trail", 6), ("Sqz", 4), ("Vol✓", 4), ("Ind", 6)]:
                table.add_column(col, justify="center" if col != "Ind" else "right", min_width=mw)

        for s in bot.iter_states():
            pos = s.position
            pnl_pct_str = pnl_usd_str = entry_str = sl_str = tp1_str = "[dim]—[/dim]"
            
            # FIX: Real-time PnL calculation
            if pos.is_open and pos.entry > 0:
                if pos.side == "LONG":
                    current_pnl = (s.price - pos.entry) / pos.entry * 100
                    current_pnl_usdt = (s.price - pos.entry) * pos.qty
                else:
                    current_pnl = (pos.entry - s.price) / pos.entry * 100
                    current_pnl_usdt = (pos.entry - s.price) * pos.qty
                    
                c = "green" if current_pnl >= 0 else "red"
                pnl_pct_str = f"[{c}]{current_pnl:+.2f}%[/]"
                pnl_usd_str = f"[{c}]{current_pnl_usdt:+.2f}[/]"
                entry_str = f"${pos.entry:,.4f}"
                sl_str = f"[red]${pos.sl_price:,.4f}[/red]"
                tp1_str = f"[green]${pos.tp1_price:,.4f}[/green]"

            rsi_c = "green" if 40 <= s.rsi_15 <= 60 else "red" if s.rsi_15 > 70 or s.rsi_15 < 30 else "yellow"
            adx_c = "green" if s.adx >= cfg.adx_min else "red"
            sc = sig_col.get(s.signal, "white")
            se = sent_col.get(s.sentiment, "yellow")
            
            # FIX: Clearer position status
            pos_s = pos.side or "[dim]—[/dim]"
            if pos.is_open and pos.trail_active:
                pos_s = f"[magenta]{pos.side}[T][/magenta]"  # [T] = trailing active
            elif pos.is_open:
                pos_s = f"[cyan]{pos.side}[/cyan]"
                
            if s.signal == "HOLD" and s.hold_reason:
                signal_cell = f"[{sc}]HOLD[/] [dim]{s.hold_reason}[/dim]"
            else:
                signal_cell = f"[{sc}]{s.signal}[/]"

            ind_age = time.time() - s.ind_ts if s.ind_ts > 0 else 9999
            ind_c = "green" if ind_age < 60 else "yellow" if ind_age < 200 else "red"
            ind_str = f"[{ind_c}]{int(ind_age)}s[/]" if s.ind_ts > 0 else "[dim]—[/dim]"
            
            # NEW: AI confidence for table
            ai_conf = "—"
            if hasattr(s, 'analysis') and s.analysis:
                if isinstance(s.analysis, dict):
                    ai_conf = str(s.analysis.get('confidence', '—'))
                else:
                    try:
                        ai_conf = self._get_ai_field(s, "confidence")
                    except Exception:
                        pass
            
            # NEW: Regime
            regime, _ = self._regime_text(s)

            row_data = [
                s.symbol, 
                f"[bold]${s.price:,.4f}[/bold]", 
                f"[{rsi_c}]{s.rsi_15:.1f}[/]",
                f"[{adx_c}]{s.adx:.0f}[/]", 
                f"[dim]{regime}[/dim]",
                f"[dim]{ai_conf}[/dim]",
                signal_cell, 
                pos_s, 
                pnl_pct_str, 
                pnl_usd_str,
            ]
            if width >= 140:
                row_data += [f"[{se}]{s.sentiment}[/]", entry_str, sl_str, tp1_str]
            if width >= 180:
                row_data += [
                    "🟣" if pos.trail_active else "·",
                    "🔵" if s.bb_squeeze else "·",
                    "✓" if s.volume_ok else "·", 
                    ind_str
                ]
            table.add_row(*row_data)
        return table

    # ── trade log panel ────────────────────────────────────────────────────
    def _trade_log_panel(self) -> Panel:
        if not self.bot.trade_log:
            return Panel("[dim]No trades yet.[/dim]", title="📋 Trade Log", border_style="dim")
        
        # FIX: Show more trades (15 instead of 8) with scroll indicator
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="dim", width=8)
        grid.add_column(style="bold cyan", width=14)
        grid.add_column(width=22)
        grid.add_column(justify="right", width=12)
        grid.add_column(justify="right", width=9)
        grid.add_column(justify="right", width=10)  # NEW: R:R column
        
        shown = self.bot.trade_log[-15:]
        for t in reversed(shown):
            # FIX: per-row error boundary — one malformed record can no longer
            # raise out of the panel and crash the whole dashboard render.
            try:
                action = t.get("action", "")
                col = ("green" if "OPEN" in action and t.get("side") == "LONG" else
                       "red" if "OPEN" in action and t.get("side") == "SHORT" else
                       "cyan" if "CLOSE" in action else "white")

                # NEW: Calculate R:R for closed trades (safe numeric parsing)
                rr_str = "—"
                if "CLOSE" in action and t.get("entry") and t.get("sl"):
                    entry = _to_float(t.get("entry"))
                    sl = _to_float(t.get("sl"))
                    exit_p = _to_float(t.get("exit"), entry)
                    risk = abs(entry - sl)
                    reward = abs(exit_p - entry)
                    rr = reward / risk if risk > 0 else 0
                    rr_str = f"{rr:.1f}x"

                # FIX: price/exit may be stored as a formatted string -> coerce first
                price_val = _to_float(t.get("price", t.get("exit", 0)))

                grid.add_row(
                    t.get("time", ""),
                    t.get("symbol", ""),
                    f"[{col}]{action} {t.get('side', '')}[/]",
                    f"${price_val:,.4f}",
                    t.get("pnl", ""),
                    rr_str,
                )
            except Exception:
                continue
        
        total = len(self.bot.trade_log)
        title = f"📋 Trade Log ({min(total, 15)}/{total} shown)"
        if total > 15:
            title += " [dim](scroll for more)[/dim]"
            
        return Panel(grid, title=title, border_style="dim")

    # ── performance panel ──────────────────────────────────────────────────
    def _performance_panel(self) -> Panel:
        """NEW: Session performance metrics."""
        bot = self.bot
        if not bot.trade_log:
            return Panel("[dim]No trades yet.[/dim]", title="📈 Performance", border_style="dim")

        # FIX: count only resolved trades, with robust PnL parsing
        pnls = [v for v in (_trade_pnl_value(t) for t in bot.trade_log) if v is not None]
        resolved = len(pnls)
        wins = sum(1 for v in pnls if v > 0)
        losses = resolved - wins
        win_rate = wins / resolved * 100 if resolved > 0 else 0

        # Calculate avg R:R (safe parsing)
        rr_values = []
        for t in bot.trade_log:
            if t.get("entry") and t.get("sl") and t.get("exit"):
                risk = abs(_to_float(t["entry"]) - _to_float(t["sl"]))
                reward = abs(_to_float(t["exit"]) - _to_float(t["entry"]))
                if risk > 0:
                    rr_values.append(reward / risk)
        avg_rr = sum(rr_values) / len(rr_values) if rr_values else 0
        
        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="bold", width=16)
        grid.add_column()
        
        # FIX: "Total Trades" now means completed trades, so it equals wins + losses
        grid.add_row("Total Trades", str(resolved))
        grid.add_row("Wins / Losses", f"[green]{wins}[/] / [red]{losses}[/]")
        grid.add_row("Win Rate", f"{win_rate:.1f}%")
        grid.add_row("Avg R:R", f"{avg_rr:.2f}x")
        grid.add_row("Daily PnL", f"[green]${bot.daily_realized_pnl:+.2f}[/]" 
                     if bot.daily_realized_pnl >= 0 else f"[red]${bot.daily_realized_pnl:+.2f}[/]")
        
        return Panel(grid, title="📈 Performance", border_style="green" if win_rate >= 50 else "yellow")

    def _analysis_section(self, width: int):
        panels = []
        for s in self.bot.iter_states():
            card = self._analysis_card(s)
            panels.append(card)
            
        n = len(panels)
        if n == 0:
            return Panel("[dim]No symbols.[/dim]")
        if width < 100:
            return panels[0]
        cols = 4 if width >= 260 else 3 if width >= 200 else 2
        rows = [Columns(panels[i:i + cols], equal=True, expand=True) for i in range(0, n, cols)]
        return RichGroup(*rows)

    def build_layout(self, balance: float) -> Layout:
        width, height = self._term_size()
        
        # FIX: Better height handling
        min_height = 20
        if height < min_height:
            height = min_height
            
        table_h = min(len(self.bot.symbol_list) + 8, height // 3)
        log_h = 10
        perf_h = 10
        
        layout = Layout()
        
        if height < table_h + log_h + perf_h + 5:
            # Compact mode
            layout.split_column(
                Layout(name="table", size=table_h),
                Layout(name="main", ratio=1),
            )
            layout["table"].update(Panel(self._summary_table(balance, width), border_style="dim"))
            layout["main"].update(self._analysis_section(width))
            return layout
            
        # Full mode with performance panel
        layout.split_column(
            Layout(name="table", size=table_h),
            Layout(name="main", ratio=2),
            Layout(name="bottom", size=log_h),
        )
        layout["table"].update(Panel(self._summary_table(balance, width), border_style="dim", padding=(0, 1)))
        
        # Split main into analysis + performance
        layout["main"].split_row(
            Layout(name="analysis", ratio=3),
            Layout(name="perf", size=25),
        )
        layout["analysis"].update(Panel(
            self._analysis_section(width),
            title="[bold]📊 AI Analysis[/bold]" if any(hasattr(s, 'analysis') and s.analysis for s in self.bot.iter_states()) else "[bold]📊 Technical Analysis[/bold]",
            border_style="dim", 
            padding=(0, 1)
        ))
        layout["perf"].update(self._performance_panel())
        layout["bottom"].update(self._trade_log_panel())
        return layout

    def print_plain_status(self, balance: float):
        """FIX: Enhanced headless output with AI info."""
        bot, cfg = self.bot, self.bot.cfg
        n_long, n_short = bot.count_open_longs(), bot.count_open_shorts()
        halt = " [HALTED]" if bot.trading_halted else ""
        
        # FIX: Header win rate over resolved trades, robust parsing (matches dashboard)
        pnls = [v for v in (_trade_pnl_value(t) for t in bot.trade_log) if v is not None]
        total = len(pnls)
        wins = sum(1 for v in pnls if v > 0)
        wr = f" WR:{wins / total * 100:.0f}%" if total > 0 else ""
        
        console.print(
            f"[bold]─ {datetime.now():%H:%M:%S} ─[/bold] "
            f"💰 ${balance:,.2f} | "
            f"PnL ${bot.daily_realized_pnl:+.2f} | "
            f"L:{n_long}/{cfg.max_open_longs} S:{n_short}/{cfg.max_open_shorts}"
            f"{wr}{halt}"
        )
        
        for s in bot.iter_states():
            pos = s.position
            tag = f" ({s.hold_reason})" if s.signal == "HOLD" and s.hold_reason else ""
            
            # FIX: Show AI info in headless mode
            ai_tag = ""
            if hasattr(s, 'analysis') and s.analysis and isinstance(s.analysis, dict):
                ai_action = s.analysis.get('action', 'WAIT')
                ai_conf = s.analysis.get('confidence', 0)
                if ai_action != 'WAIT' and ai_conf > 0:
                    ai_tag = f" [AI:{ai_action}:{ai_conf}%]"
            
            # Real-time PnL
            if pos.is_open and pos.entry > 0:
                if pos.side == "LONG":
                    pnl = (s.price - pos.entry) / pos.entry * 100
                else:
                    pnl = (pos.entry - s.price) / pos.entry * 100
                posinfo = f"{pos.side} {pnl:+.2f}%"
                if pos.trail_active:
                    posinfo += "[T]"
            else:
                posinfo = "—"
                
            pstr = f"${s.price:,.4f}" if s.price else "  (loading)"
            regime, _ = self._regime_text(s)
            
            console.print(
                f"  {s.symbol:<13} {pstr:>14}  RSI{s.rsi_15:5.1f}  "
                f"ADX{s.adx:4.0f}  {regime:<10}  {s.signal}{tag:<10}  "
                f"{posinfo}{ai_tag}"
            )