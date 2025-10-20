import logging
import time
import numpy as np 
from utils.IB_connector import IBapi


def execute_trade(app : IBapi, action, alloc, price, prev_value, cash, position, max_allocation, transaction_cost, asset_name):
    execute_trade = False
    shares = 0    
    if action == 1:  # BUY
        current_allocation = (position * price) / prev_value
        allocation_left = max(0, max_allocation - current_allocation)
        allocation_left = min(allocation_left, alloc)

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
                        f"Buying {shares} of asset {asset_name} at price {price:.3f} "
                        f"with invest amount {invest_amount:.3f}, allocation {allocation_left:.3f} and cost {cost:.3f}, cash now {cash:.3f}"
                    )
            app.buy_market(asset_name, qty=shares)
            alloc = allocation_left
            execute_trade = True


    elif action == 2:  # SELL
        shares = position
        if shares > 0:
            revenue = shares * price
            position -= shares
            cash += revenue
            logging.info(
                        f"Selling {shares} of asset {asset_name} at price {price:.3f} "
                        f"with revenue {revenue:.3f}, cash now {cash:.3f}"
                    )
            app.sell_market(asset_name, qty=shares)
            alloc = 1
            execute_trade = True
    
    
    return {
        "action": "BUY" if action == 1 else "SELL" if action == 2 else "HOLD",
        "allocation": alloc,
        "price": price,
        "position": shares,
        "asset_name": asset_name,
        "executed" : execute_trade,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
