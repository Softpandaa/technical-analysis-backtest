"""Parameters of the technical analysis backtests."""

# Sample ----------------------------------------------------------------------
UNIVERSE = [  # (yfinance ticker, SigTech contract code); only the ticker is read
    ("^AEX", "EO"), ("XU030.IS", "A5"), ("^BFX", "BE"), ("^FCHI", "CF"),
    ("2823.HK", "XU"), ("^GDAXI", "GX"), ("EXSG.DE", "DSD"), ("YM=F", "DM"),
    ("^IXIC", "NQ"), ("^GSPC", "ES"), ("XLY", "IXY"), ("XLE", "IXP"),
    ("XLF", "IXA"), ("XLU", "IXI"), ("^STOXX50E", "VG"), ("EXV5.DE", "EB"),
    ("EXX1.DE", "CA"), ("EXV6.DE", "DA"), ("EXV7.DE", "CU"), ("EXV1.DE", "AW"),
]
START = "2015-01-01"
END = "2026-09-16"             # exclusive, the date of the published run
AUTO_ADJUST = False            # the strategies read an Adj Close column

# Rules -----------------------------------------------------------------------
EMA_SPANS = (21, 50)           # fast, slow
MACD_SPANS = (12, 26, 9)       # fast, slow, signal
RSI_PERIOD = 6
RSI_MA_PERIOD = 14
ADX_PERIOD = 6
ADX_MA_PERIOD = 14
BB_PERIOD = 20
BB_INNER = 1.0                 # standard deviations
BB_OUTER = 2.0                 # standard deviations
STOP_LOSS = 0.05               # rule 1 stop, a fraction below the close

# Backtest --------------------------------------------------------------------
TRADING_DAYS = 252             # per year
INITIAL_CAPITAL = 100_000.0
