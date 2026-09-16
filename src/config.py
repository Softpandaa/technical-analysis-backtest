"""Parameters for the technical analysis backtests. Every constant is declared here once."""
from datetime import date

# Universe used by every strategy. Each entry pairs the yfinance ticker with the
# contract code the original work used on SigTech; only the ticker is read here.
UNIVERSE = [
    ("^AEX", "EO"), ("XU030.IS", "A5"), ("^BFX", "BE"), ("^FCHI", "CF"),
    ("2823.HK", "XU"), ("^GDAXI", "GX"), ("EXSG.DE", "DSD"), ("YM=F", "DM"),
    ("^IXIC", "NQ"), ("^GSPC", "ES"), ("XLY", "IXY"), ("XLE", "IXP"),
    ("XLF", "IXA"), ("XLU", "IXI"), ("^STOXX50E", "VG"), ("EXV5.DE", "EB"),
    ("EXX1.DE", "CA"), ("EXV6.DE", "DA"), ("EXV7.DE", "CU"), ("EXV1.DE", "AW"),
]

START = "2015-01-01"
END = date.today().isoformat()
AUTO_ADJUST = False            # the strategies read an Adj Close column

EMA_SPANS = (21, 50)           # fast, slow
MACD_SPANS = (12, 26, 9)       # fast, slow, signal
RSI_PERIOD = 6
RSI_MA_PERIOD = 14
ADX_PERIOD = 6
ADX_MA_PERIOD = 14
BB_PERIOD = 20                 # Bollinger window
BB_INNER = 1.0                 # deviations of the inner band
BB_OUTER = 2.0                 # deviations of the outer band

TRADING_DAYS = 252
