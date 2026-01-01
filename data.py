# get finance data and caching (first versoin checks for chached files, second version adds data starting from date in cached file if available)
import pandas as pd
import numpy as np
import yfinance as yf

from test_subjects import *


# get open, high, low, close, volume data for x ticker symbol over period
def get_ticker_yf(ticker_symbol, period):
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period=period)


# cache to db from csv
def cache_to_db(csv_filename):
    pass


# saves stock data of all ticker symbols from a csv file into db
def save_list_to_db(csv_filename):
    pass




# get info for a stock from db
def get_stock_info(ticker_symbol):
    pass



# get info for all stocks in db minus chose one
def get_all_stocks_info_minus(ticker_symbol):
    pass