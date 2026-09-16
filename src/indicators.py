"""Indicators, transcribed from the notebooks without changing any definition.

Three of them depart from the textbook form. The departures are the working
code's own and are preserved here rather than corrected.

    rsi     averages gains and losses with a simple moving average, not with
            Wilder's smoothing, so this is Cutler's RSI and will not agree with
            the conventional form.
    macd    returns the histogram, the MACD line minus its signal line, which
            the notebooks store in a column named MACD.
    vwap    accumulates from the first row of the sample rather than resetting,
            so on daily data it is an expanding mean price, not a session VWAP.
"""
import numpy as np
import pandas as pd

import config


def ema(close, span):
    return close.ewm(span=span, adjust=False).mean()


def macd(close, spans=config.MACD_SPANS):
    """The MACD histogram, as the notebooks define it."""
    fast_span, slow_span, signal_span = spans
    line = ema(close, fast_span) - ema(close, slow_span)
    return line - ema(line, signal_span)


def rsi(close, period=config.RSI_PERIOD):
    delta = close.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    average_gain = up.rolling(period).mean()
    average_loss = abs(down.rolling(period).mean())
    return 100 - (100 / (1 + average_gain / average_loss))


def vwap(close, volume):
    return (close * volume).cumsum() / volume.cumsum()


def bollinger(close, period=config.BB_PERIOD, deviations=1.0):
    """Simple moving average and its band at the given number of deviations."""
    middle = close.rolling(period).mean()
    spread = close.rolling(period).std(ddof=0) * deviations
    return middle - spread, middle, middle + spread


def adx(high, low, close, period=config.ADX_PERIOD):
    """Wilder's directional index."""
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
