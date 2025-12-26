# get stocks to be used to learn and test models, given to data.py to fetch and cache data of listed stocks
# adds stock names and ticker symbols into test_subjects.csv

import pandas as pd
import numpy as np
import requests
import json
import os
from dotenv import load_dotenv


load_dotenv() # load environment variables from .env file

API_NINJAS_KEY = os.getenv("API_NINJAS_KEY")
if not API_NINJAS_KEY:
    raise RuntimeError("Missing API_NINJAS_KEY. Put it in a .env file.")

# gets list of s&p 500 stocks (~top 500 biggest)
def get_snp500_stocks_api():

    #accesses api for s&p 500 stock list using api key (get request)
    api_url = "https://api.api-ninjas.com/v1/sp500"
    response = requests.get(api_url, headers={'X-Api-Key': API_NINJAS_KEY})

    # puts data into dataframe
    if response.status_code == requests.codes.ok:
        data = response.json()
        df = pd.DataFrame(data)
        return df
    
# gets list of snp500 stocks (~top 500 biggest)
def get_snp500_stocks():
    #s&p 500 stock list using csv of IVV (ishares snp 500 etf) holdings
    df = pd.read_csv("https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund", skiprows=9, skipfooter=10, engine='python')
    return df

# gets list of russell 1000 stocks (~top 1000 biggest)
def get_russell1000_stocks():
    #russell 1000 stock list using csv of IWB (ishares russell 1000 etf) holdings
    df = pd.read_csv("https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund", skiprows=9, skipfooter=10, engine='python')
    return df


# gets list of russell 2000 stocks (~top 2000 smallest)
def get_russell2000_stocks():
    #russell 2000 stock list using csv of IWM (ishares russell 2000 etf) holdings
    df = pd.read_csv("https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund", skiprows=9, skipfooter=10, engine='python')
    return df

# gets list of russell 3000 stocks (~top 3000 biggest + smallest)
def get_russell3000_stocks():
    #russell 3000 stock list using csv of IWV (ishares russell 3000 etf) holdings
    df = pd.read_csv("https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund", skiprows=9, skipfooter=10, engine='python')
    return df


# normalizes ticker symbols (some have dots, dashes, etc)
def normalize_ticker(ticker):
    pass


# merges two stock lists, removes duplicates, saves to test_subjects.csv
def merge_stock_lists(list1, list2):
    pass


    