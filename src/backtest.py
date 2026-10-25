"""Prices, trade lists, daily equity and measures, with zero commission (gross of costs)."""
import numpy as np
import pandas as pd
import yfinance as yf

import config


def prices(ticker, start=config.START, end=config.END, auto_adjust=config.AUTO_ADJUST):
    """Daily OHLCV for one ticker, indexed by date."""
    df = yf.download(ticker, start, end, auto_adjust=auto_adjust, progress=False)
    if isinstance(df.columns[0], tuple):  # yfinance returns a MultiIndex for one ticker
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def trades(close, position, stop_loss):
    """Long trades: open on position +1, close on -1 or below the stop (stop_loss a fraction)."""
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
            if price[i] < stop or pos[i] == -1:
                closes.append(close.index[i])
                close_px.append(price[i])
                in_market = False
    if len(opens) != len(closes):
        closes.append(close.index[-1])
        close_px.append(price[-1])
    return pd.DataFrame({"opened": opens, "closed": closes,
                         "open_price": open_px, "close_price": close_px})


def _size(history, initial_capital):
    """Whole share sizing, all of the account in one position, compounded."""
    capital, shares, pnl, curve = float(initial_capital), [], [], []
    for _, row in history.iterrows():
        n = int(capital / row["open_price"])
        gain = row["side"] * n * (row["close_price"] - row["open_price"])
        capital += gain
        shares.append(n)
        pnl.append(gain)
        curve.append(capital)
    history["shares"], history["pnl"], history["equity"] = shares, pnl, curve
    history["pnl_pct"] = 100 * history["pnl"] / (history["equity"] - history["pnl"])
    return history


def equity(history, initial_capital):
    """A long only trade list, sized with side +1."""
    out = history.copy()
    out.insert(2, "side", 1.0)
    return _size(out, initial_capital)


def run(close, position, stop_loss, initial_capital=config.INITIAL_CAPITAL):
    """Trade list for a long only state crossover with a stop."""
    return equity(trades(close, position, stop_loss), initial_capital)


def positions_from_trades(close, history):
    """Daily position of a long only trade list, held from the entry close to the exit close."""
    position = pd.Series(0.0, index=close.index)
    for _, row in history.iterrows():
        i, j = close.index.get_indexer([row["opened"], row["closed"]])
        position.iloc[i:j] = 1.0
    return position


def daily_equity(close, position, initial_capital=config.INITIAL_CAPITAL):
    """Daily mark to market: the position of day t-1 earns the return of day t."""
    returns = close.pct_change().fillna(0.0)
    strategy = position.shift(1).fillna(0.0) * returns
    return initial_capital * (1.0 + strategy).cumprod(), strategy


def drawdown(equity):
    return equity / equity.cummax() - 1.0


def measures(close, position, history, initial_capital=config.INITIAL_CAPITAL):
    """Every figure of the results tables, from the daily curve and the trade list."""
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
        out["profit_factor"] = (float(gains.sum() / abs(losses.sum())) if len(losses)
                                else float("inf"))
        out["mean_trade_pct"] = history["pnl_pct"].mean()
        out["sd_trade_pct"] = history["pnl_pct"].std(ddof=1)
    else:
        out.update({"win_rate": float("nan"), "profit_factor": float("nan"),
                    "mean_trade_pct": float("nan"), "sd_trade_pct": float("nan")})
    return out


def from_positions(close, position, initial_capital=config.INITIAL_CAPITAL):
    """Trade list implied by a daily position series (rules 2 to 4)."""
    pos = np.asarray(position, float)
    price = close.to_numpy(float)
    opens, closes, open_px, close_px, sides = [], [], [], [], []
    current = 0.0
    for i in range(len(pos)):
        if pos[i] != current:
            if current != 0.0:
                closes.append(close.index[i])
                close_px.append(price[i])
            if pos[i] != 0.0:
                opens.append(close.index[i])
                open_px.append(price[i])
                sides.append(pos[i])
            current = pos[i]
    if len(opens) > len(closes):
        closes.append(close.index[-1])
        close_px.append(price[-1])
    history = pd.DataFrame({"opened": opens, "closed": closes, "side": sides,
                            "open_price": open_px, "close_price": close_px})
    return _size(history, initial_capital)
