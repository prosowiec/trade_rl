import pandas as pd
from source.database import read_stock_data
from source.IB_connector import retrive_market_data


def get_training_set_from_IB(ticker : str) -> pd.DataFrame:
    training_set_tickers = retrive_market_data([ticker], duration = "9 m", time_interval = "15 mins")
    training_set = training_set_tickers[ticker]
    training_set['Volume'] = training_set['Volume'].astype(float)
    training_set['Date'] = pd.to_datetime(training_set['Date'].str.replace(' US/Eastern',''),format="%Y%m%d %H:%M:%S")
    training_set['ticker'] = ticker
    return training_set


def get_recent_data(tickers):
    training_set_tickers = retrive_market_data(tickers, duration = "7 d", time_interval = "15 mins", sleep_time=2)
    data = pd.DataFrame()
    for ticker in tickers:
        training_set = training_set_tickers[ticker]
        training_set['Volume'] = training_set['Volume'].astype(float)
        training_set['Date'] = pd.to_datetime(training_set['Date'].str.replace(' US/Eastern',''),format="%Y%m%d %H:%M:%S")

        temp = pd.DataFrame(training_set['Close'].copy()).rename(columns={'Close': ticker})

        data = data.reset_index(drop=True)    
        data = pd.concat([data, temp], axis=1)

    
    return data