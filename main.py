import logging
import time
import numpy as np
import threading
import random
from agents.traderModel import get_trading_desk
from manager_training import AgentPortfolio
from utils.dataOps import get_recent_data, get_observation
from utils.IB_connector import retrieve_positions, retrieve_account_and_portfolio, IBapi
from utils.database import save_trade_to_db
from tickers import Tickers
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

class NoSendingIBMessagesFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not (
            "SENDING placeOrder" in msg or
            "SENDING reqHistoricalData" in msg or
            "ANSWER updateAccountTime" in msg 
        )

logger = logging.getLogger()
logger.addFilter(NoSendingIBMessagesFilter())

ib_loggers = [
    logging.getLogger("ibapi"),
    logging.getLogger("ibapi.client"),
    logging.getLogger("ibapi.wrapper"),
    logging.getLogger("ibapi.connection"),
]

for ib_logger in ib_loggers:
    ib_logger.addFilter(NoSendingIBMessagesFilter())



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
        shares = position # np.floor(min(position * alloc, position))
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


def main(app : IBapi):
    tickers = Tickers().TICKERS_penny
    trading_desk = get_trading_desk(tickers)
    
    logging.info("Loaded trading desk with agents for tickers: " + ", ".join(tickers))
    portfolio_manager = AgentPortfolio(input_dim=96, action_dim=len(tickers))
    portfolio_manager.load_agent()
    logging.info("Loaded portfolio manager agent.")
    
    
    WINDOW_SIZE = 96

    while True:
        data = get_recent_data(app,tickers)        
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
            if portfolio.empty:
                value = 0
            else:
                ticker_position = portfolio[portfolio['symbol'] == ticker]
                value = ticker_position['position'].iloc[0] if not ticker_position.empty else 0   
            positions.append(int(value))     


        current_state = get_observation(data[-WINDOW_SIZE:], WINDOW_SIZE, traders_actions, positions, float(cash), len(tickers))
        
        action_allocation_percentages = np.array([portfolio_manager.get_action_target(current_state)]).flatten()

        for i, key in enumerate(trading_desk.keys()):
            logging.info(f"Trader for {key} action: {traders_actions[i]} | Allocation: {action_allocation_percentages[i]:.3f}")
            current_price = data[key].values[-1]
            
            trade_data = execute_trade(app, traders_actions[i], action_allocation_percentages[i], current_price, float(current_value), float(cash), 
                          positions[i], 0.5, 1, key)
            
            save_trade_to_db(trade_data)

        # 15 minut = 900 sekund
        time.sleep(60 * 15)

if __name__ == "__main__":
    app = IBapi()
    app.connect("127.0.0.1", 4002, clientId=random.randint(1, 9999)) #7497

    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    time.sleep(2)

    try:
        main(app)
    except Exception as e:
        logging.exception(f"Error in main(): {e}")

    finally:
        # --- Bezpieczne zakończenie ---
        logging.info("Ending connection with IB brokest...")
        try:
            app.disconnect()
        except Exception as e:
            logging.warning(f"Failed to end connection with IB: {e}")

        logging.info("Program end.")
        sys.exit(0)