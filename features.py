#indicators and window building
from data import *
import pandas as pd
import numpy as np
import sqlite3


# gets data from db and puts it into a df
def db_to_df(db_file_name):

    with sqlite3.connect(db_file_name) as conn:
        cursor = conn.cursor()
        query = '''
            SELECT *
            FROM stocks_table
            ORDER BY ticker_symbol, date ASC
           
        '''
        cursor.execute(query)
        rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])
    return df



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