import time
import numpy as np
import threading
import random
from apscheduler.schedulers.background import BackgroundScheduler
from agents.traderModel import get_trading_desk
from pytz import timezone
from manager_training import AgentPortfolio
from utils.trade import execute_trade
from utils.dataOps import get_recent_data, get_observation
from utils.IB_connector import retrieve_positions, retrieve_account_and_portfolio, IBapi
from utils.database import save_trade_to_db
from tickers import Tickers
import logging
from utils.logger import start_logger
import sys

start_logger()

def trading_job(app, trading_desk, portfolio_manager, WINDOW_SIZE):
    """The actual trading logic - called by scheduler"""
    try:
        tickers = list(trading_desk.keys())
        
        data = get_recent_data(app, tickers)        
        positions = retrieve_positions(app)
        portfolio, account = retrieve_account_and_portfolio(app)
        current_value = float(account['NetLiquidation'])
        cash = float(account['AvailableFunds'])
        app.cancel_all_open_orders()
        traders_actions = []
        positions = []
        for i, ticker in enumerate(tickers):
            curr_trader = trading_desk[ticker]
            trader_input = data[ticker].values
            traders_actions.append(curr_trader.get_action(trader_input[-WINDOW_SIZE:], target_model=True))
            
            if portfolio.empty:
                value = 0
            else:
                ticker_position = portfolio[portfolio['symbol'] == ticker]
                value = ticker_position['position'].iloc[0] if not ticker_position.empty else 0   
            positions.append(int(value))     

        current_state = get_observation(data[-WINDOW_SIZE:], WINDOW_SIZE, traders_actions, positions, float(cash), len(tickers))
        action_allocation_percentages = np.array([portfolio_manager.get_action_target(current_state)]).flatten()

        for i, key in enumerate(tickers):
            logging.info(f"Trader for {key} action: {traders_actions[i]} | Allocation: {action_allocation_percentages[i]:.3f}")
            current_price = data[key].values[-1]
            
            trade_data = execute_trade(app, traders_actions[i], action_allocation_percentages[i], current_price, float(current_value), float(cash), 
                          positions[i], 0.5, 1, key)
            
            save_trade_to_db(trade_data)
            
        logging.info("Trading job completed successfully")
        
    except Exception as e:
        logging.exception(f"Error in trading_job: {e}")


def main(app: IBapi, tickers_group = 'PENNY', dashboard_enabled = False):
    if not dashboard_enabled:
        tickers = Tickers().get_tickers(tickers_group)
    else:
        ...
        
    trading_desk = get_trading_desk(tickers)
    
    logging.info("Loaded trading desk with agents for tickers: " + ", ".join(tickers))
    portfolio_manager = AgentPortfolio(tickers, input_dim=96, action_dim=len(tickers))
    portfolio_manager.load_agent()
    logging.info("Loaded portfolio manager agent.")
    
    WINDOW_SIZE = 96

    # Create scheduler
    scheduler = BackgroundScheduler()
    
    # Create scheduler with ET timezone
    et = timezone('US/Eastern')
    scheduler = BackgroundScheduler(timezone=et)
    
    # Schedule job to run at :00, :15, :30, :45 between 9:30 AM and 4:00 PM ET
    scheduler.add_job(
        trading_job,
        'cron',
        hour='9-15',
        minute='0,15,30,45',
        start_date='2024-01-01 09:30:00',
        args=(app, trading_desk, portfolio_manager, WINDOW_SIZE),
        id='trading_job',
        name='Trading execution every 15 minutes (9:30 AM - 4:00 PM ET)',
        timezone=et
    )    
    scheduler.start()
    logging.info("="*15 + "Scheduler started at {}".format(time.strftime("%Y-%m-%d %H:%M:%S")) + "="*15 )
    
    try:
        # Keep the scheduler running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Scheduler interrupted by user")
        scheduler.shutdown()


if __name__ == "__main__":
    time.sleep(5)
    app = IBapi()
    logging.info("=" * 60)
    logging.info("Connecting to IB broker...")
    app.connect("ib-gateway", 4004, clientId=random.randint(1, 9999))

    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    time.sleep(2)

    try:
        main(app)
    except Exception as e:
        logging.exception(f"Error in main(): {e}")

    finally:
        logging.info("Ending connection with IB broker...")
        try:
            app.disconnect()
        except Exception as e:
            logging.warning(f"Failed to end connection with IB: {e}")

        logging.info("Program end.")
        sys.exit(0)