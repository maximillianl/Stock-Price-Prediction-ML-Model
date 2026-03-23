# backtesting
from IDTW import *
import numpy as np
import pandas as pd
from numba import njit


def backtest_compare_DTW(stock, test_stocks, w = 20, features_compared = FEATURES_DEFAULT, top_n = 10):
    

    # test_tickers = ['GOOG', 'AAPL', 'MSFT', 'AMZN', 'META']
    # small_df = test_stocks[test_stocks['Ticker'].isin(test_tickers)]

    stock_df = test_stocks[test_stocks['Ticker'] == stock]
    target_stock = stock_df.iloc[-w*2:-w]
    
    
    actual_outcome = stock_df.iloc[-w:]
    # temp to test shortlist
    # test_stocks = small_df.drop(target_stock.index)
    test_stocks = test_stocks.drop(target_stock.index).drop(actual_outcome.index)

    target_values = target_stock[features_compared].values

    target_z = z_score(target_values)

    results = []
    window_step = w//4

    for ticker in test_stocks['Ticker'].unique():
        # df of just current ticker
        ticker_df = test_stocks[test_stocks['Ticker'] == ticker]
        print(ticker, "======================================================================")
        
        # slides windows through ticker at each window_step
        for start in range(0, len(ticker_df) - w, window_step):
            comparitor = ticker_df.iloc[start:start + w]
            comparitor_values = comparitor[features_compared].values
            if np.isnan(comparitor_values).any():
                continue
            comparitor_z = z_score(comparitor_values)
            
            print(type(target_z), type(comparitor_z), type(DTW))
            
            DTW_score = DTW(target_z, comparitor_z, w // 4)
            results.append({'Ticker': ticker, 'start_date': ticker_df['Date'].iloc[start], 'end_date': ticker_df['Date'].iloc[start + w - 1], 'DTW_score': DTW_score})
            print(DTW_score)


    results.sort(key=lambda x: x['DTW_score'])
    return results[:top_n]