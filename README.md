# Stock-Price-Prediction-ML-Model

**Note:** This is a work in progress and will be updated as improvements are made. I'm actively exploring different time series models and machine learning techniques to improve forecasting accuracy.

This project explores machine learning and statistical modeling techniques to predict stock prices using technical indicators and time series forecasting methods. Data is pulled directly from Yahoo Finance via the `yfinance` API.

---

## Development Log

### Day 1:
- Pulled historical stock data using `yfinance`
- Engineered Multiple Features:
  - Moving averages (SMA, EMA)
  - RSI
  - MACD
  - Volatility
  - Bollinger bands
  - ATR
- Built a linear regression model using selected features
- Framework works, but predictions are inconsistent, not time dependent

### Day 2:
- Researched time series forecasting methods:
  - ARIMA / SARIMA / SARIMAX
  - LSTM / RNN
  - Prophet
  - XGBoost
  - VAR
- Began ARIMA implementation
  - Differenced data and checked stationarity (ADF test)
  - Fitted ARIMA(1,1,0) and other variants
  - Predictions not ideal yet — model often produces a flat or trending line (likely tuning issue or misaligned forecast)

---

Upcoming Goals:
- Improve ARIMA accuracy
- Implement SARIMAX (using more variables)
- Explore LSTM-based time series modeling
- Forecast visualization improvements

---

## Technologies Used
- Python
- Pandas, NumPy
- scikit-learn
- statsmodels
- matplotlib
- yfinance
