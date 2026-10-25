"""Signals and positions of the four rules."""
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
    """Trailing stop at width percent (one value or one per day); signal +1 long, -1 short."""
    n = len(close)
    long_, short_ = np.zeros(n, bool), np.zeros(n, bool)
    level = np.zeros(n)
    price, sig = close.to_numpy(float), np.asarray(signal, float)
    w = np.broadcast_to(np.asarray(width, float), n)
    below, above = 1 - w / 100, 1 + w / 100
    for i in range(n):
        if sig[i] == 1:
            long_[i:], short_[i:] = True, False
            level[i:] = price[i] * below[i]
        elif sig[i] == -1:
            short_[i:], long_[i:] = True, False
            level[i:] = price[i] * above[i]
        else:
            if long_[i] and price[i] * below[i] > level[i]:
                level[i:] = price[i] * below[i]
            elif short_[i] and price[i] * above[i] < level[i]:
                level[i:] = price[i] * above[i]
            if long_[i] and price[i] < level[i]:
                long_[i:] = False
            elif short_[i] and price[i] > level[i]:
                short_[i:] = False
    return pd.DataFrame({"trailing_stop": level, "long": long_, "short": short_},
                        index=close.index)


def _held(entry, exit_):
    """One from an entry to the next exit; on a day with both, the exit wins."""
    entry, exit_ = np.asarray(entry), np.asarray(exit_)
    held = np.zeros(len(entry))
    for i in range(len(entry)):
        if entry[i] == 1:
            held[i:] = 1
        if exit_[i] == 1:
            held[i:] = 0
    return held


def carry(long_entry, long_exit, short_entry, short_exit):
    """Long leg minus short leg, each held from its entry to its exit."""
    return pd.Series(_held(long_entry, long_exit) - _held(short_entry, short_exit),
                     index=long_entry.index)


def macd_signal(close, spans=config.MACD_SPANS):
    """+1 where the histogram crosses up through zero (g_t = +2), -1 where it crosses down."""
    g = np.sign(ind.macd(close, spans)).diff()
    return np.sign(g).where(g.abs() == 2, 0.0)


def confirmation(close, high, low, volume,
                 rsi_period=config.RSI_PERIOD, rsi_ma=config.RSI_MA_PERIOD,
                 adx_period=config.ADX_PERIOD, adx_ma=config.ADX_MA_PERIOD,
                 spans=config.MACD_SPANS):
    """RSI, VWAP and ADX agreement, gated on MACD and the day's return; doji veto dropped."""
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
    long_entry = (momentum == 1) & (change == 1) & (total == 3)
    short_entry = (momentum == -1) & (change == -1) & (total == -3)
    long_exit = (c_vwap == -1) | (c_adx == -1)
    short_exit = (c_vwap == 1) | (c_adx == 1)
    return carry(long_entry, long_exit, short_entry, short_exit)


def _crossover(a, b):
    """True on the days a crosses above b, that is, b crosses below a."""
    return (a > b) & (a.shift(1) <= b.shift(1))


def _entries(arm, trigger):
    """One on the first trigger day once armed; entering disarms."""
    entry, armed = np.zeros(len(arm), int), False
    for i in range(len(arm)):
        if arm.iloc[i]:
            armed = True
        if armed and trigger.iloc[i]:
            entry[i], armed = 1, False
    return pd.Series(entry, index=arm.index)


def bollinger_reversion(close, period=config.BB_PERIOD,
                        inner=config.BB_INNER, outer=config.BB_OUTER):
    """Double Bollinger rule; the source states no exit, so the middle band exit is a choice."""
    lower_outer, _, upper_outer = ind.bollinger(close, period, outer)
    lower_inner, middle, upper_inner = ind.bollinger(close, period, inner)
    long_entry = _entries(close < lower_outer, _crossover(close, lower_inner))
    short_entry = _entries(close > upper_outer, _crossover(upper_inner, close))
    long_exit, short_exit = _crossover(close, middle), _crossover(middle, close)
    return carry(long_entry, long_exit, short_entry, short_exit)


def stop_width(close):
    """Trailing stop width w_t of rule 2 in percent, from the daily returns up to day t."""
    r = (close / close.shift(1) - 1) * 100
    return abs(r.expanding().mean()) + r.expanding().std()
