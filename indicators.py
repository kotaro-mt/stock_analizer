"""Technical indicators: MA, RSI, MACD, volume, Ichimoku Kinko Hyo.

Pure pandas/numpy. No TA-Lib dependency. Every function takes a price
``pd.Series`` (or OHLCV DataFrame) and returns values aligned to the
same index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------
def sma(prices: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return prices.rolling(window, min_periods=window).mean()


def ema(prices: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return prices.ewm(span=span, adjust=False).mean()


# ---------------------------------------------------------------------------
# RSI (Wilder)
# ---------------------------------------------------------------------------
def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing (equivalent to EMA with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(50.0)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------
def macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, histogram — returned as a DataFrame."""
    macd_line = ema(prices, fast) - ema(prices, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": hist}
    )


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------
def volume_ma(volume: pd.Series, window: int = 20) -> pd.Series:
    return volume.rolling(window, min_periods=window).mean()


# ---------------------------------------------------------------------------
# Ichimoku Kinko Hyo (一目均衡表)
# ---------------------------------------------------------------------------
def ichimoku(
    df: pd.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
) -> pd.DataFrame:
    """Ichimoku Kinko Hyo.

    Returns a DataFrame with columns:
      - ``tenkan``   転換線   : (H9 + L9) / 2
      - ``kijun``    基準線   : (H26 + L26) / 2
      - ``span_a``   先行スパンA: ((tenkan + kijun) / 2) shifted forward 26
      - ``span_b``   先行スパンB: ((H52 + L52) / 2) shifted forward 26
      - ``chikou``   遅行スパン: close shifted back 26

    Notes:
      The ``span_a``/``span_b`` values at any given index are already
      the values that should be *plotted* at that index (i.e. the cloud
      for today is computed from data 26 bars ago).
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2.0
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2.0
    span_a = ((tenkan_sen + kijun_sen) / 2.0).shift(kijun)
    span_b = (
        (high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2.0
    ).shift(kijun)
    chikou = close.shift(-kijun)

    return pd.DataFrame(
        {
            "tenkan": tenkan_sen,
            "kijun": kijun_sen,
            "span_a": span_a,
            "span_b": span_b,
            "chikou": chikou,
        }
    )


# ---------------------------------------------------------------------------
# Convenience: all indicators at once
# ---------------------------------------------------------------------------
def compute_all(df: pd.DataFrame) -> dict:
    """Compute the standard indicator bundle for a chart display."""
    close = df["Close"]
    volume = df["Volume"]
    return {
        "sma5": sma(close, 5),
        "sma25": sma(close, 25),
        "sma75": sma(close, 75),
        "sma200": sma(close, 200),
        "volume_ma20": volume_ma(volume, 20),
        "rsi14": rsi(close, 14),
        "macd": macd(close),
        "ichimoku": ichimoku(df),
    }
