# stock_forecasting.py
# In-progress model for stock price prediction using ARIMA + linear regression
# Code includes test models, debug prints, and data exploration

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from statsmodels.tsa.arima.model import ARIMA

#test
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


def get_stock_data(ticker_symbol, period):
    
    #create ticker object
    ticker = yf.Ticker(ticker_symbol)
    
    #fetch historical market data
    historical_data = ticker.history(period = period)
    
    stock_data = pd.DataFrame(historical_data)
    # print(stock_data.tail(20))
    return stock_data

#simple moving average, exponential moving average
def moving_avgs(df):
    
    #calculates simple moving average for each 50 day window
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

    return df


#rsi <30 = oversold, >70 = overbought
def rsi(df,period):
    #difference in close price today-yesterday
    delta = df["Close"].diff()
    delta.dropna()

    change_up,change_down = delta.copy(),delta.copy()

    # leaves up pos, down neg, otherwise 0
    change_up[change_up<0] = 0
    change_down[change_down>0] = 0

    #SMA based averages
    avg_gain = change_up.rolling(period).mean()
    avg_loss = change_down.rolling(period).mean().abs()
    df["RSI"] = 100 - (100 /( 1 + (avg_gain/avg_loss)))
    
    return df
    
#moving average convergence divergence (momentum)
def MACD(df):
    #12 day ema - 26 day ema
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean()-df["Close"].ewm(span=26, adjust=False).mean()

    #signal line
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    # df.plot.line(y=["MACD","MACD_Signal"], color=['red','blue'], marker='o')
    # plt.grid(True)
    # plt.show()

    return df

#relative volatility
def bollinger_bands(df,period):
    sma = df["Close"].rolling(window=period).mean()
    std_dev = df["Close"].rolling(window=period).std()
    df["Upper_BB"] = sma + (2 * std_dev)
    df["Lower_BB"] = sma - (2 * std_dev)
    return df

#raw volatility
def ATR(df,period):
    df['High-Low'] = df['High'] - df['Low']
    df['High-Close'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-Close'] = abs(df['Low'] - df['Close'].shift(1))
    
    df['TR'] = df[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    
    df['ATR'] = df['TR'].rolling(window=period).mean()
    df.drop(columns=['High-Low', 'High-Close', 'Low-Close', 'TR'], )
    
    return df



    
#price prediction (Linear Regression) (non time dependent, basically random numbers)
def price_prediction_LR(df, period):
    # print("[][][][][][]")
    # print(df.tail(20))
    
    df = df[:len(df)-5]
    # df.dropna()
    
    
    #independent variables depending on prediction horizon
    if period <= 5:
        X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'ATR']]
    elif period <= 30:
        X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'ATR']]
    else:
        X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_200', 'EMA_200', 'RSI', 'MACD', 'MACD_Signal', 'ATR']]
        
    X.dropna(inplace = True)    
        
        
        
        
    #dependent variables
    y = df['Close'].shift(-period)
    
    
    y.dropna(inplace = True)   
    
    
    
    
    #make X an y the same length
    X = X.iloc[:len(y)]
    y = y.iloc[:len(X)]
    

    # Splitting dataset into test/train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    # Regressor model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # print(X.tail)
    # print(y.tail)
    
    
    # # Prediction result
    y_pred_test = regressor.predict(X_test)     # predicted value of y_test
    y_pred_train = regressor.predict(X_train)   # predicted value of y_train
    print(y_pred_test, "++++", y_pred_train)
    
    # # Evaluate test set performance
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    # # Evaluate training set performance
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    print(f"Test Set - MAE: {mae_test}, MSE: {mse_test}, R2: {r2_test}")
    print(f"Train Set - MAE: {mae_train}, MSE: {mse_train}, R2: {r2_train}")







def PLOT(df):
    df.plot.line(y='Close', color='green')
    plt.grid(True)
    plt.show()



#===== work in progress ======
#time series price prediction (ARIMA)
def price_prediction_ARIMA(df, period):
    # df = df[:len(df)-5]
    close = df['Close'].dropna()
    # print(df.isna().sum(), "=================", df.shape)

    # #draws the graph of the stock
    # df.plot.line(y='Close', color='green')
    # plt.grid(True)
    # plt.show()
    


    #train is all but last 30 days, test is last 30 days
    train = close[:len(df)-30].copy()
    test = close[len(df)-30:].copy()
    
    #transform to stationary: differencing
    train_diff = train.diff().dropna()
    # PLOT(test)
    

    #acf and pacf plot for testing
    # acf_diff = plot_acf(train_diff)
    # pacf_diff = plot_pacf(train_diff)
    # plt.show()

    
    #p value test
    # adf_test = adfuller(train_diff)
    # print(f'p-value: {adf_test[1]}')
    # print(f'p-value: {adf_test[1]:.2000f}')

    #ARIMA d=1 to use differenced


    #fit and determine model parameters
    model = ARIMA(train, order=(1,1,0))
    model_fit = model.fit()
    # print(model_fit.summary())

    # model = ARIMA(train, order=(0,1,1))
    # model_fit = model.fit()
    # print(model_fit.summary())

    # random walk
    # model = ARIMA(train, order=(0,1,0))
    # model_fit = model.fit()
    # print(model_fit.summary())



    # #plot residuals and density
    # residuals = model_fit.resid[1:]
    # #1 row 2 columns
    # figure, axes = plt.subplots(1,2)
    # #should look like white noise
    # residuals.plot(title='Residuals', ax=axes[0])
    # residuals.plot(title='Density', kind='kde',ax=axes[1])
    # plt.show()


    # #make sure it shows few to no spikes (residuals are white noise)
    # acf_res = plot_acf(residuals)
    # pacf_res = plot_pacf(residuals)
    # plt.show()


    forecast_test = model_fit.forecast(len(test))

    #[None]*len(train) creates a list of None values for all training rows + appends values after training set ends
    result_df = pd.DataFrame({'Close': close,
                             'model': [None]*len(train) + list(forecast_test)})

    result_df.plot(title='Actual vs Predicted Price')
    plt.show()





stock = get_stock_data("BMI", "2y")
moving_avgs(stock)
rsi(stock,14)
MACD(stock)
bollinger_bands(stock, 20)
ATR(stock,14)
# print(stock.shape)
# print(stock.isna().sum())
# print(stock.head(20))

price_prediction_ARIMA(stock,5)

# price_prediction_LR(stock,5)
