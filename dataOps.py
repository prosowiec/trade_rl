import pandas as pd
from source.database import read_stock_data
from source.IB_connector import retrive_market_data


def get_training_set_from_IB(ticker : str) -> pd.DataFrame:
    training_set_aapl = retrive_market_data([ticker], duration = "9 m", time_interval = "15 mins")
    training_set = training_set_aapl[ticker]
    training_set['Volume'] = training_set['Volume'].astype(float)
    training_set['Date'] = pd.to_datetime(training_set['Date'].str.replace(' US/Eastern',''),format="%Y%m%d %H:%M:%S")
    training_set['ticker'] = ticker
    return training_set


