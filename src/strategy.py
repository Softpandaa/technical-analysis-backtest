"""Signal rules, transcribed from the original notebooks.

Two crossover conventions are in use and both are kept. state_crossover holds an
indicator of state, one while the fast line is above the slow line and zero
otherwise, and trades its first difference. macd_signal takes the sign of the
first difference of the sign of the spread, which fires plus or minus one on the
crossing day only.
"""
import numpy as np
import pandas as pd

import config
import indicators as ind


def state_crossover(fast, slow):
    """One while fast is above slow, zero otherwise, and its first difference."""
    signal = pd.Series(np.where(fast > slow, 1, 0), index=fast.index)
    return signal, signal.diff()


def ema_state(close, spans=config.EMA_SPANS):
    fast, slow = (ind.ema(close, s) for s in spans)
    return state_crossover(fast, slow)


def trailing_stop(close, signal, width):
    """Percentage trailing stop, transcribed from get_trailing_stop.

    width is the stop distance in per cent. signal is plus one to go long, minus
    one to go short and zero otherwise.
    """
    n = len(close)
    long_, short_ = np.zeros(n, bool), np.zeros(n, bool)
    level, exit_ = np.zeros(n), np.zeros(n)
    price, sig = close.to_numpy(float), np.asarray(signal, float)
    for i in range(n):
        if sig[i] == 1:
            long_[i:], short_[i:] = True, False
            level[i:] = price[i] * (1 - width / 100)
        elif sig[i] == -1:
            short_[i:], long_[i:] = True, False
            level[i:] = price[i] * (1 + width / 100)
        else:
            if long_[i] and price[i] * (1 - width / 100) > level[i]:
                level[i:] = price[i] * (1 - width / 100)
            elif short_[i] and price[i] * (1 + width / 100) < level[i]:
                level[i:] = price[i] * (1 + width / 100)
            if long_[i] and price[i] < level[i]:
                exit_[i], long_[i:] = -1, False
            elif short_[i] and price[i] > level[i]:
                exit_[i], short_[i:] = 1, False
    return pd.DataFrame({"trailing_stop": level, "long": long_, "short": short_},
                        index=close.index)


def carry(long_entry, long_exit, short_entry, short_exit):
    """Hold a position from entry until its exit, transcribed from TA_function.

    The two tests are independent rather than exclusive, so on a day that both
    enters and exits the exit wins, which is the original's ordering.
    """
    n = len(long_entry)
    long_, short_ = np.zeros(n), np.zeros(n)
    le, lx = np.asarray(long_entry), np.asarray(long_exit)
    se, sx = np.asarray(short_entry), np.asarray(short_exit)
    for i in range(n):
        if le[i] == 1:
            long_[i:] = 1
        if lx[i] == 1:
            long_[i:] = 0
    for i in range(n):
        if se[i] == 1:
            short_[i:] = -1
        if sx[i] == 1:
            short_[i:] = 0
    return pd.Series(long_ + short_, index=getattr(long_entry, "index", None))


def macd_signal(close, spans=config.MACD_SPANS):
    """Plus one where the histogram crosses up through zero, minus one down."""
    histogram = ind.macd(close, spans)
    return np.sign(np.sign(histogram).diff())


def confirmation(close, high, low, volume,
                 rsi_period=config.RSI_PERIOD, rsi_ma=config.RSI_MA_PERIOD,
                 adx_period=config.ADX_PERIOD, adx_ma=config.ADX_MA_PERIOD,
                 spans=config.MACD_SPANS):
    """RSI, VWAP and ADX agreement, gated on the MACD histogram and the day's return.

    The original also vetoed any day carrying a doji, a dragonfly doji or a
    gravestone doji. That veto is dropped, so this rule takes trades the
    notebooks suppressed and its results do not reproduce theirs.
    """
    rsi = ind.rsi(close, rsi_period)
    adx = ind.adx(high, low, close, adx_period)
    benchmark = adx.rolling(adx_ma).mean() - adx.rolling(adx_ma).std()
    histogram = ind.macd(close, spans)

    c_rsi = np.sign(rsi - rsi.rolling(rsi_ma).mean())
    c_vwap = np.sign(close - ind.vwap(close, volume))
    c_adx = np.sign(adx - benchmark)
    total = c_rsi + c_vwap + c_adx

    change = np.sign(close / close.shift(1) - 1)
    momentum = np.sign(histogram.diff())
    long_entry = pd.Series(np.where((momentum == 1) & (change == 1) & (total == 3), 1, 0),
                           index=close.index)
    short_entry = pd.Series(np.where((momentum == -1) & (change == -1) & (total == -3), 1, 0),
                            index=close.index)
    long_exit = np.where(c_vwap == -1, 1, np.where(c_adx == -1, 1, 0))
    short_exit = np.where(c_vwap == 1, 1, np.where(c_adx == 1, 1, 0))
    return carry(long_entry, long_exit, short_entry, short_exit)


def bollinger_reversion(close, period=config.BB_PERIOD,
                        inner=config.BB_INNER, outer=config.BB_OUTER):
    """Range bounded double Bollinger rule, constructed rather than transcribed.

    The source notebook computes the bands and a pair of potential signals and
    stops there, with no entry, exit or backtest. The rule below is built from
    its markdown: arm once price closes outside the outer band, enter on the
    close back through the inner band, exit at the middle band. The exit is a
    choice, not something the source states.
    """
    lower_outer, _, upper_outer = ind.bollinger(close, period, outer)
    lower_inner, middle, upper_inner = ind.bollinger(close, period, inner)

    below = close < lower_outer
    above = close > upper_outer
    cross_up = (close > lower_inner) & (close.shift(1) <= lower_inner.shift(1))
    cross_down = (close < upper_inner) & (close.shift(1) >= upper_inner.shift(1))

    n = len(close)
    le, se = np.zeros(n, int), np.zeros(n, int)
    armed_long = armed_short = False
    for i in range(n):
        if below.iloc[i]:
            armed_long = True
        if above.iloc[i]:
            armed_short = True
        if armed_long and cross_up.iloc[i]:
            le[i], armed_long = 1, False
        if armed_short and cross_down.iloc[i]:
            se[i], armed_short = 1, False
    long_entry = pd.Series(le, index=close.index)
    short_entry = pd.Series(se, index=close.index)
    long_exit = ((close > middle) & (close.shift(1) <= middle.shift(1))).astype(int)
    short_exit = ((close < middle) & (close.shift(1) >= middle.shift(1))).astype(int)
    return carry(long_entry, long_exit, short_entry, short_exit)


def stop_width(close):
    """Trailing stop width in per cent, transcribed from the MACD notebooks.

    The width is the absolute mean plus the standard deviation of the daily
    return, measured on close.shift(-1) / close - 1 over the whole sample. Both
    the forward shift and the use of the full sample look past the day the stop
    is set. This is the original's definition and is kept.
    """
    forward = (close.shift(-1) / close - 1) * 100
    return abs(forward.mean()) + forward.std(ddof=1)
