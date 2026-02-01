#indicators and window building
from data import *
import pandas as pd
import numpy as np
import sqlite3


# loads single stock data from db to dataframe
def load_stock_from_db(ticker_symbol):
    pass



# gets log returns of stock prices (log of daily returns)

# get stock is a dataframe with ohlcv data from db
# use data from 'Close' and 'Open' columns
def log_returns(stock, window=20):
    pass
    # log ( close / open )



#gets upper wick (log distance from max of open/close to high)
def upper_wick(stock, window=20):
    pass
    # log ( high / max(open, close) )





#gets lower wick (log distance from min of open/close to low)
def lower_wick(stock, window=20):
    pass
    # log ( min(open, close) / low )



#gets art% + percentile rank (volatility)
def volatility(stock, window=20):
    pass




#gets volume / 20-day avg volume (relative volume rvol)
def rvol(stock, window=20):
    pass



# gets % deviation from EMA (trend strength)
def ema_deviation(stock, window=20):
    pass



#gets donchian channel percentile ()
def donchian_channel_percentile(stock, window=20):
    pass



#gets rsi (mean reversion momentum)
def rsi(stock, window=20):
    pass