import logging
import time
import numpy as np
from source.IB_connector import retrieve_positions, retrieve_account_and_portfolio, IBapi
import threading
import random
from trader import DQNAgent
from manager import AgentPortfolio
from manager_env import PortfolioEnv
from dataOps import get_recent_data, get_observation

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


def execute_trade(app : IBapi, action, alloc, price, prev_value, cash, position, max_allocation, transaction_cost, asset_name):
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
            app.buy_market(asset_name, qty=shares)


    elif action == 2:  # SELL
        shares_to_sell = np.floor(min(position * alloc, position))
        if shares_to_sell > 0:
            revenue = shares_to_sell * price
            position -= shares_to_sell
            cash += revenue
            logging.info(
                        f"Selling {shares_to_sell} of asset {asset_name} at price {price} "
                        f"with revenue {revenue}, cash now {cash}"
                    )
            app.sell_market(asset_name, qty=shares_to_sell)
    


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
    
    app = IBapi()
    app.connect("127.0.0.1", 7497, clientId=random.randint(1, 9999))

    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    time.sleep(1)

    
    WINDOW_SIZE = 96

    while True:
        data = get_recent_data(tickers)        
        positions = retrieve_positions(app)
        portfolio, account = retrieve_account_and_portfolio(app)
        current_value = float(account['NetLiquidation'])
        cash = float(account['AvailableFunds'])

        traders_actions = []
        positions = []
        for i, ticker in enumerate(trading_desk.keys()):
            curr_trader = trading_desk[ticker]
            trader_iput = data[ticker].values
            # get trader action
            traders_actions.append(curr_trader.get_action(trader_iput[-WINDOW_SIZE:], target_model = True))
            
            #get current positions
            ticker_position = portfolio[portfolio['symbol'] == ticker]
            value = ticker_position['position'].iloc[0] if not ticker_position.empty else 0   
            positions.append(int(value))     


        # get portfolio manager action
        current_state = get_observation(data[-WINDOW_SIZE:], WINDOW_SIZE, traders_actions, positions, float(cash), len(tickers))
        
        action_allocation_percentages = np.array([portfolio_manager.get_action_target(current_state)]).flatten()
        
        # place orders
        for i, key in enumerate(trading_desk.keys()):
            logging.info(f"Trader for {key} action: {traders_actions[i]} | Allocation: {action_allocation_percentages[i]:.3f}")
            current_price = data[key].values[-1]
            
            # #get funds from 
            # portfolio, account = retrieve_account_and_portfolio(app)
            # current_value = account['NetLiquidation']
            # cash = account['AvailableFunds']

            execute_trade(app, traders_actions[i], action_allocation_percentages[i], current_price, float(current_value), float(cash), 
                          positions[i], 0.3, 1, key)

        # 15 minut = 900 sekund
        time.sleep(60 * 15)

if __name__ == "__main__":
    main()
