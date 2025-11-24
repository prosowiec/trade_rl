import pandas as pd
import numpy as np
from utils.IB_connector import retrive_market_data, IBapi
import threading


def get_training_set_from_IB(app, ticker : str) -> pd.DataFrame:
    training_set_tickers = retrive_market_data(app, [ticker], duration = "9 m", time_interval = "15 mins")
    training_set = training_set_tickers[ticker]
    training_set['Volume'] = training_set['Volume'].astype(float)
    training_set['Date'] = pd.to_datetime(training_set['Date'].str.replace(' US/Eastern',''),format="%Y%m%d %H:%M:%S")
    training_set['ticker'] = ticker
    return training_set


def get_recent_data(app, tickers, duration = "7 d", sleep_time=3):
    training_set_tickers = retrive_market_data(app, tickers, duration = duration, time_interval = "15 mins", sleep_time=sleep_time)
    data = pd.DataFrame()
    for ticker in tickers:
        training_set = training_set_tickers[ticker]
        training_set['Volume'] = training_set['Volume'].astype(float)
        training_set['Date'] = pd.to_datetime(training_set['Date'].str.replace(' US/Eastern',''),format="%Y%m%d %H:%M:%S")

        temp = pd.DataFrame(training_set['Close'].copy()).rename(columns={'Close': ticker})

        data = data.reset_index(drop=True)    
        data = pd.concat([data, temp], axis=1)

    
    return data

def get_recent_data_for_UI(tickers, duration = "7 d", host="ib-gateway", port=4004, client_id=1):
    
    app = IBapi()
    app.connect(host, port, client_id)

    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()

    training_set_tickers = retrive_market_data(app, tickers, duration = duration, time_interval = "15 mins", sleep_time=1)
    data = pd.DataFrame()  # pusty DataFrame do łączenia

    for ticker in tickers:
        training_set = training_set_tickers[ticker]
        
        training_set['Volume'] = training_set['Volume'].astype(float)
        training_set['Date'] = pd.to_datetime(
            training_set['Date'].str.replace(' US/Eastern', ''), 
            format="%Y%m%d %H:%M:%S"
        )

        temp = training_set[['Date', 'Close']].copy().rename(columns={'Close': ticker})
        temp = temp.set_index('Date')

        if data.empty:
            data = temp
        else:
            # Łączenie po indeksie (Date)
            data = data.join(temp, how='outer')

    data = data.sort_index()
    app.disconnect()

    return data

def get_observation(close_data, window_size, trader_action, position, cash, n_assets):
    curr_prices = close_data.iloc[-window_size:].values  # (window, n_assets)

    # normalizacja cen
    min_vals = close_data.min().values
    max_vals = close_data.max().values
    norm_prices = (curr_prices - min_vals) / (max_vals - min_vals + 1e-8)

    # normalizacja akcji (BUY=1, SELL=-1, HOLD=0)
    norm_actions = trader_action.copy()
    for i in range(n_assets):
        if trader_action[i] == 1:
            norm_actions[i] = 1.0
        elif trader_action[i] == 2:
            norm_actions[i] = -1.0
        else:
            norm_actions[i] = 0.0
            
    norm_actions = np.array(norm_actions)[:, np.newaxis]

    last_prices = close_data.iloc[-1].values
    asset_values = position * last_prices
    total_value = cash + np.sum(asset_values)
    portfolio_shares = asset_values / (total_value + 1e-8)
    cash_share = cash / (total_value + 1e-8)

    portfolio_shares = portfolio_shares[:, np.newaxis]
    cash_share = cash_share * np.ones((n_assets, 1)) 

    obs = np.concatenate([
        norm_prices.T, 
        norm_actions, 
        portfolio_shares, 
        cash_share
    ], axis=1)

    return np.round(obs, 4)
