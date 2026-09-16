"""The trade list backtest shared by the crossover strategies.

Transcribed from long_EMA_Backtest and short_EMA_Backtest with one change. The
notebooks charged a commission whose formula differed between the first trade
and every later one, and whose stated rate was ignored after the first trade.
Commission is zero here, as the scope of this repository carries no trading
costs, so position size and profit are gross of costs throughout.
"""
import numpy as np
import pandas as pd

import config


def trades(close, position, stop_loss):
    """Entry and exit dates and prices from a state crossover and a stop.

    position is the first difference of the state, plus one to open and minus one
    to close. stop_loss is a fraction, so 0.05 sits five per cent below.
    """
    price = close.to_numpy(float)
    pos = np.asarray(position, float)
    in_market, stop = False, 0.0
    entry_price = 0.0
    opens, closes, open_px, close_px = [], [], [], []
    for i in range(len(price)):
        if not in_market:
            if pos[i] == 1:
                opens.append(close.index[i])
                open_px.append(price[i])
                entry_price = price[i]
                stop = price[i] * (1 - stop_loss)
                in_market = True
        else:
            if price[i] > entry_price:
                stop = price[i] * (1 - stop_loss)
            if price[i] < stop:
                closes.append(close.index[i])
                close_px.append(price[i])
                in_market = False
            elif pos[i] == -1:
                closes.append(close.index[i])
                close_px.append(price[i])
                in_market = False
    if len(opens) != len(closes):
        closes.append(close.index[-1])
        close_px.append(price[-1])
    return pd.DataFrame({"opened": opens, "closed": closes,
                         "open_price": open_px, "close_price": close_px})


def equity(history, initial_capital):
    """Whole share sizing, all of the account in one position, compounded."""
    capital = float(initial_capital)
    shares, pnl, curve = [], [], []
    for _, row in history.iterrows():
        n = int(capital / row["open_price"])
        gain = n * (row["close_price"] - row["open_price"])
        capital += gain
        shares.append(n)
        pnl.append(gain)
        curve.append(capital)
    out = history.copy()
    out.insert(2, "side", 1.0)
    out["shares"], out["pnl"], out["equity"] = shares, pnl, curve
    out["pnl_pct"] = 100 * out["pnl"] / (out["equity"] - out["pnl"])
    return out


def run(close, position, stop_loss, initial_capital=100000.0):
    """Trade list for a long only state crossover with a stop."""
    return equity(trades(close, position, stop_loss), initial_capital)


def positions_from_trades(close, history):
    """Daily position implied by a long only trade list."""
    position = pd.Series(0.0, index=close.index)
    for _, row in history.iterrows():
        position.loc[row["opened"]:row["closed"]] = 1.0
    return position


def daily_equity(close, position, initial_capital=100000.0):
    """Mark the position to market each day.

    The position of day t-1 earns the return of day t, so nothing is traded on
    information from the day it is measured.
    """
    returns = close.pct_change().fillna(0.0)
    strategy = position.shift(1).fillna(0.0) * returns
    return initial_capital * (1.0 + strategy).cumprod(), strategy


def drawdown(equity):
    return equity / equity.cummax() - 1.0


def measures(close, position, history, initial_capital=100000.0):
    """Every figure the notebooks report, from the daily curve and the trade list."""
    equity, strategy = daily_equity(close, position, initial_capital)
    years = (close.index[-1] - close.index[0]).days / 365.25
    volatility = strategy.std(ddof=1) * np.sqrt(config.TRADING_DAYS)

    out = {"trades": len(history),
           "time_in_market": float((position != 0).mean()),
           "strategy_cagr": ((equity.iloc[-1] / initial_capital) ** (1 / years) - 1) * 100,
           "hold_cagr": ((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1) * 100,
           "max_drawdown": float(drawdown(equity).min()) * 100,
           "sharpe": float(strategy.mean() * config.TRADING_DAYS / volatility)
           if volatility > 0 else float("nan")}
    out["cagr_alpha"] = out["strategy_cagr"] - out["hold_cagr"]

    if len(history):
        pnl = history["pnl"]
        gains, losses = pnl[pnl > 0], pnl[pnl < 0]
        out["win_rate"] = 100 * (pnl > 0).mean()
        out["profit_factor"] = float(gains.sum() / abs(losses.sum())) if len(losses) else float("inf")
        out["mean_trade_pct"] = history["pnl_pct"].mean()
        out["sd_trade_pct"] = history["pnl_pct"].std(ddof=1)
    else:
        out.update({"win_rate": float("nan"), "profit_factor": float("nan"),
                    "mean_trade_pct": float("nan"), "sd_trade_pct": float("nan")})
    return out


def from_positions(close, position, initial_capital=100000.0):
    """Trade list implied by a daily position series, for the carry forward rules."""
    pos = np.asarray(position, float)
    price = close.to_numpy(float)
    opens, closes, open_px, close_px, sides = [], [], [], [], []
    current = 0.0
    for i in range(len(pos)):
        if pos[i] != current:
            if current != 0.0:
                closes.append(close.index[i]); close_px.append(price[i])
            if pos[i] != 0.0:
                opens.append(close.index[i]); open_px.append(price[i]); sides.append(pos[i])
            current = pos[i]
    if len(opens) > len(closes):
        closes.append(close.index[-1]); close_px.append(price[-1])
    history = pd.DataFrame({"opened": opens, "closed": closes, "side": sides,
                            "open_price": open_px, "close_price": close_px})
    capital, shares, pnl, curve = float(initial_capital), [], [], []
    for _, row in history.iterrows():
        n = int(capital / row["open_price"])
        gain = row["side"] * n * (row["close_price"] - row["open_price"])
        capital += gain
        shares.append(n); pnl.append(gain); curve.append(capital)
    history["shares"], history["pnl"], history["equity"] = shares, pnl, curve
    history["pnl_pct"] = 100 * history["pnl"] / (history["equity"] - history["pnl"])
    return history
