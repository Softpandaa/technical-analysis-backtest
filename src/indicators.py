"""Indicators of the four rules."""
import numpy as np
import pandas as pd

import config


def ema(close, span):
    return close.ewm(span=span, adjust=False).mean()


def macd(close, spans=config.MACD_SPANS):
    """The MACD histogram: the MACD line minus its signal line."""
    fast_span, slow_span, signal_span = spans
    line = ema(close, fast_span) - ema(close, slow_span)
    return line - ema(line, signal_span)


def rsi(close, period=config.RSI_PERIOD):
    """Cutler's RSI: simple averages of gains and losses, not Wilder's smoothing."""
    delta = close.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    average_gain = up.rolling(period).mean()
    average_loss = abs(down.rolling(period).mean())
    return 100 - (100 / (1 + average_gain / average_loss))


def vwap(close, volume):
    """Volume weighted close accumulated from the first row, not a session VWAP."""
    return (close * volume).cumsum() / volume.cumsum()


def bollinger(close, period=config.BB_PERIOD, deviations=1.0):
    """Simple moving average and its band at the given number of deviations."""
    middle = close.rolling(period).mean()
    spread = close.rolling(period).std(ddof=0) * deviations
    return middle - spread, middle, middle + spread


def adx(high, low, close, period=config.ADX_PERIOD):
    """ADX with the DIs from rolling sums, not Wilder's smoothing."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).sum()
    plus_di = 100 * plus_dm.rolling(period).sum() / atr
    minus_di = 100 * minus_dm.rolling(period).sum() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, adjust=False).mean()
