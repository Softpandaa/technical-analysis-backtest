"""Price download. The strategies read an Adj Close column, so adjustment stays off."""
import yfinance as yf

import config


def prices(ticker, start=config.START, end=config.END, auto_adjust=config.AUTO_ADJUST):
    """Daily OHLCV for one ticker, indexed by date."""
    df = yf.download(ticker, start, end, auto_adjust=auto_adjust, progress=False)
    if isinstance(df.columns[0], tuple):          # yfinance returns a MultiIndex for one ticker
        df.columns = df.columns.get_level_values(0)
    return df.dropna()
