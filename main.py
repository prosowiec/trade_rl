import logging
import time
import numpy as np
from source.IB_connector import retrieve_positions, retrieve_account_and_portfolio

from trader import DQNAgent
from manager import AgentPortfolio
from manager_env import PortfolioEnv
from dataOps import get_recent_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)


def get_trading_desk(tickers):
    trading_desk = {}
    for ticker in tickers:
        trader = DQNAgent(ticker)
        trader.load_dqn_agent()
        
        trading_desk[ticker] = trader

    return trading_desk

def get_observation(close_data, window_size, trader_action, position, cash, n_assets):
    """
    Args:
        window_size (int): liczba kroków w oknie danych cenowych
        close_data (pd.DataFrame): ceny zamknięcia (shape: [time, n_assets])
        trader_action (np.ndarray): akcje tradera dla każdego aktywa
        position (np.ndarray): aktualne pozycje (liczba akcji dla aktywów)
        cash (float): dostępna gotówka
        n_assets (int): liczba aktywów
    """

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

def execute_trade(action, alloc, price, prev_value, cash, position, max_allocation, transaction_cost, asset_name):
    if action == 1:  # BUY
        current_allocation = (position * price) / prev_value
        allocation_left = max(0, max_allocation - current_allocation)

        invest_amount = cash * allocation_left
        invest_amount = min(invest_amount + transaction_cost, cash)
        shares = np.floor(invest_amount / price)

        max_shares_to_buy = max(np.floor((cash - transaction_cost) / price), 0)
        shares = min(shares, max_shares_to_buy)
        cost = shares * price + transaction_cost

        if shares > 0:
            position += shares
            cash -= cost
            logging.info(
                        f"Buying {shares} of asset {asset_name} at price {price} "
                        f"with invest amount {invest_amount} and cost {cost}, cash now {cash}"
                    )


    elif action == -1:  # SELL
        shares_to_sell = np.floor(min(position * alloc, position))
        if shares_to_sell > 0:
            revenue = shares_to_sell * price
            position -= shares_to_sell
            cash += revenue
            logging.info(
                        f"Selling {shares_to_sell} of asset {asset_name} at price {price} "
                        f"with revenue {revenue}, cash now {cash}"
                    )     


    return cash, position


def main():
    tickers =  ['NVDA', 'MSFT', 'AAPL', 'GOOG', 'AMZN',
                'META', 'AVGO', 'TSLA', 'JPM',
                'WMT', 'V', 'ORCL', 'LLY', 'NFLX',
                'MA', 'XOM', 'JNJ'  
        ]
    trading_desk = get_trading_desk(tickers)
    
    logging.info("Loaded trading desk with agents for tickers: " + ", ".join(tickers))
    portfolio_manager = AgentPortfolio(input_dim=96, action_dim=len(tickers))
    portfolio_manager.load_agent()
    logging.info("Loaded portfolio manager agent.")
    
    WINDOW_SIZE = 96

    while True:
        data = get_recent_data(tickers)        

        traders_actions = []
        for i,key in enumerate(trading_desk.keys()):
            curr_trader = trading_desk[key]
            trader_iput = data[key].values
            traders_actions.append(curr_trader.get_action(trader_iput[-WINDOW_SIZE:], target_model = True))

        positions = retrieve_positions()
        portfolio, account = retrieve_account_and_portfolio()
        current_value = account['NetLiquidation']
        cash = account['AvailableFunds']

        positions = []
        
        for ticker in tickers:
            ticker_position = portfolio[portfolio['symbol'] == ticker]

            value = ticker_position['position'].iloc[0] if not ticker_position.empty else 0   
            positions.append(int(value))     
            
        current_state = get_observation(data[-WINDOW_SIZE:], WINDOW_SIZE, traders_actions, positions, float(cash), len(tickers))
        action_allocation_percentages = portfolio_manager.get_action_target(current_state)
        action_allocation_percentages = np.array([action_allocation_percentages]).flatten()
        action = {
            'trader' : traders_actions,
            'portfolio_manager': action_allocation_percentages
        }
        
        orders = {}
        for i, key in enumerate(trading_desk.keys()):
            logging.info(f"Trader for {key} action: {traders_actions[i]} | Allocation: {action_allocation_percentages[i]:.3f}")
            orders[key] = {
                'action': traders_actions[i],
                'shares': action_allocation_percentages[i]
            }
            current_price = data[key].values[-1]
            
            print(execute_trade(traders_actions[i], action_allocation_percentages[i], current_price, float(current_value), float(cash), 
                          positions[i], 0.3, 1, key))

        # 15 minut = 900 sekund
        time.sleep(15 * 60)

if __name__ == "__main__":
    main()
