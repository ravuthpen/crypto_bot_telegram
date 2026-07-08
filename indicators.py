"""
indicators.py — self-contained technical indicators (pure pandas/numpy)
=======================================================================
Drop-in replacement for the handful of pandas_ta functions the bot uses,
so we don't depend on pandas_ta (which is unmaintained and breaks builds).

Each function mirrors the pandas_ta call signature AND output shape used in
bot.py, so `import indicators as ta` works as a direct swap:

    import indicators as ta
    df["RSI"]   = ta.rsi(df["Close"], length=14)
    df["SMA20"] = ta.sma(df["Close"], length=20)
    df["EMA50"] = ta.ema(df["Close"], length=50)
    df["ATR"]   = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    adx_df      = ta.adx(df["High"], df["Low"], df["Close"], length=14)   # col: ADX_14
    macd_df     = ta.macd(df["Close"], fast=12, slow=26, signal=9)        # cols: MACD_12_26_9, MACDs_12_26_9
    bb          = ta.bbands(df["Close"], length=20, std=2.0)              # cols: BBL/BBM/BBU/BBB/BBP_20_2.0
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def sma(close: pd.Series, length: int = 20) -> pd.Series:
    return close.rolling(length).mean()


def ema(close: pd.Series, length: int = 50) -> pd.Series:
    # adjust=False matches the standard recursive EMA used by most TA libs
    return close.ewm(span=length, adjust=False).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0.0)
    loss  = -delta.clip(upper=0.0)
    # Wilder's smoothing (RMA) == ewm with alpha = 1/length
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # No downside in the window → RSI 100. Crucially, only force 100 where we
    # actually have data; the warm-up rows (avg_gain/avg_loss still NaN) must
    # stay NaN so a caller's "is this fresh?" / fillna-fallback logic works,
    # instead of being silently reported as a maxed-out 100.
    out = out.mask((avg_loss == 0.0) & avg_gain.notna(), 100.0)
    return out


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    # Wilder's RMA
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
    """
    Returns a DataFrame with at least column 'ADX_{length}' (bot only reads that).
    Also provides DMP/DMN for completeness.
    """
    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm  = pd.Series(plus_dm,  index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr  = _true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    plus_di  = 100.0 * (plus_dm.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()  / atr_.replace(0.0, np.nan))
    minus_di = 100.0 * (minus_dm.ewm(alpha=1 / length, min_periods=length, adjust=False).mean() / atr_.replace(0.0, np.nan))

    dx  = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_series = dx.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    return pd.DataFrame({
        f"ADX_{length}":  adx_series,
        f"DMP_{length}":  plus_di,
        f"DMN_{length}":  minus_di,
    })


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Returns columns MACD_{f}_{s}_{sig}, MACDh_..., MACDs_... (bot reads MACD_ and MACDs_)."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist        = macd_line - signal_line
    suffix = f"{fast}_{slow}_{signal}"
    return pd.DataFrame({
        f"MACD_{suffix}":  macd_line,
        f"MACDh_{suffix}": hist,
        f"MACDs_{suffix}": signal_line,
    })


def bbands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands. Column names match pandas_ta's float-suffix style
    (e.g. BBL_20_2.0) so the bot's prefix-based lookup finds them.
    Includes BBB (bandwidth) and BBP (%B).
    """
    mid   = close.rolling(length).mean()
    sd    = close.rolling(length).std(ddof=0)
    upper = mid + std * sd
    lower = mid - std * sd
    bandwidth = (upper - lower) / mid.replace(0.0, np.nan)
    percent_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    suffix = f"{length}_{std}"
    return pd.DataFrame({
        f"BBL_{suffix}": lower,
        f"BBM_{suffix}": mid,
        f"BBU_{suffix}": upper,
        f"BBB_{suffix}": bandwidth,
        f"BBP_{suffix}": percent_b,
    })
