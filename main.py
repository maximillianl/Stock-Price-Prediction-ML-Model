import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def get_stock_data(ticker_symbol, period):
    
    #create ticker object
    ticker = yf.Ticker(ticker_symbol)
    
    #fetch historical market data
    historical_data = ticker.history(period = period)
    
    stock_data = pd.DataFrame(historical_data)
    print(stock_data.tail(20))
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
    delta.dropna(inplace=True)

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

#volatility
def bollinger_bands(df,period):
    sma = df["Close"].rolling(window=period).mean()
    std_dev = df["Close"].rolling(window=period).std()
    df["Upper_BB"] = sma + (2 * std_dev)
    df["Lower_BB"] = sma - (2 * std_dev)
    return df

def ATR(df,period):
    df['High-Low'] = df['High'] - df['Low']
    df['High-Close'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-Close'] = abs(df['Low'] - df['Close'].shift(1))
    
    df['TR'] = df[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    
    df['ATR'] = df['TR'].rolling(window=period).mean()
    df.drop(columns=['High-Low', 'High-Close', 'Low-Close', 'TR'], inplace=True)
    
    return df



    
#price prediction after period
def price_prediction(df, period):
    print("[][][][][][]")
    print(df.tail(20))
    
    df = df[:len(df)-5]
    # df.dropna(inplace=True)
    
    
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

    print(X.tail)
    print(y.tail)
    






   
    
    
    
    
    
    # # Prediction result
    # y_pred_test = regressor.predict(X_test)     # predicted value of y_test
    # y_pred_train = regressor.predict(X_train)   # predicted value of y_train
    
    # # Evaluate test set performance
    # mae_test = mean_absolute_error(y_test, y_pred_test)
    # mse_test = mean_squared_error(y_test, y_pred_test)
    # r2_test = r2_score(y_test, y_pred_test)

    # # Evaluate training set performance
    # mae_train = mean_absolute_error(y_train, y_pred_train)
    # mse_train = mean_squared_error(y_train, y_pred_train)
    # r2_train = r2_score(y_train, y_pred_train)

    # print(f"Test Set - MAE: {mae_test}, MSE: {mse_test}, R2: {r2_test}")
    # print(f"Train Set - MAE: {mae_train}, MSE: {mse_train}, R2: {r2_train}")
    
    
    
    
    
    
# precent change prediction
def percent_prediction(df, period):
   
    df = df[:len(df)-5]
    #independent variables depending on prediction horizon
    if period <= 5:
        X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'ATR']]
    elif period <= 30:
        X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'ATR']]
    else:
        X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_200', 'EMA_200', 'RSI', 'MACD', 'MACD_Signal', 'ATR']]
        
        
    #dependent variables
    y = (df['Close'].shift(-period) - df['Close']) / df['Close'] * 100
    
    #make X an y the same length
    X = X.iloc[:len(y)]
    
    
    print(X, ("+"), y)
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Splitting dataset into test/train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    # Regressor model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    
    # Prediction result
    y_pred_test = regressor.predict(X_test)     # predicted value of y_test
    y_pred_train = regressor.predict(X_train)   # predicted value of y_train
    
    print(y_pred_test, ("+", y_pred_train))





stock = get_stock_data("IBM", "1y")
moving_avgs(stock)
rsi(stock,14)
MACD(stock)
bollinger_bands(stock, 20)
ATR(stock,14)
print(stock.shape)
print(stock.isna().sum())

price_prediction(stock,5)
