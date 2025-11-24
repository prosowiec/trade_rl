import numpy as np
from tqdm import tqdm
import pandas as pd

from agents.traderModel import DQNAgent
from agents.managerModel import AgentPortfolio
from utils.database import read_stock_data
from eval.eval_portfolio import evaluate_steps_portfolio, render_portfolio_summary
from agent_env.manager_env import PortfolioEnv
from tickers import Tickers
from utils.state_managment import set_seed
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)


def train_porfolio_manager(env, trading_desk, episode):
    episode_reward = 0
    step = 1

    
    current_state = env.reset()
    done = False
    while not done:
        
        action_allocation_percentages = portfolio_manager.get_action(current_state)
        
        traders_actions = []
        hist_data = env.get_price_window()
        for i,key in enumerate(trading_desk.keys()):
            curr_trader = trading_desk[key]
            traders_actions.append(curr_trader.get_action(hist_data[i,:], target_model = True))

        action = {
            'trader' : traders_actions,
            'portfolio_manager': np.array([action_allocation_percentages]).flatten()
        }
        if not env.current_step % 500:
            print(np.array([action_allocation_percentages]).flatten())
        new_state, reward, done, info = env.step(action)
        episode_reward += reward
        
        portfolio_manager.update_replay_memory((current_state, action, reward, new_state, done))
        
        portfolio_manager.train()
        
        current_state = new_state
        step += 1
 
    portfolio_manager.save_agent()
    logging.info(f"Saved model | Episode: {episode} | Total Reward: {env.total_portfolio:.2f}")
    
    
    return episode_reward




if __name__ == "__main__":
    #set_seed(42, derterministic=False)

    tickers = Tickers().TRASH_TICKERS
    trading_desk = {}
    data = pd.DataFrame()
    min_size = 9999999
    for ticker in tickers:
        trader = DQNAgent(ticker)
        trader.load_dqn_agent()
        trading_desk[ticker] = trader

        train_data, valid_data, test_data = read_stock_data(ticker)
        training_set = pd.concat([train_data, valid_data, test_data])

        min_size = min(min_size, len(training_set))

        temp = pd.DataFrame(training_set['close'].copy()).rename(columns={'close': ticker})

        temp = temp[:min_size].reset_index(drop=True)
        data = data[:min_size].reset_index(drop=True)    
        data = pd.concat([data[:min_size], temp[:min_size]], axis=1)
        
    reward_all = []
    evaluate_revards = []
    portfolio_manager = AgentPortfolio(tickers, input_dim=96, action_dim=len(tickers))


    data_split = int(len(data)  * 0.8)

    train_data = data[:data_split]
    valid_data = data[data_split:]

    WINDOW_SIZE = 96
    env = PortfolioEnv(train_data, window_size=WINDOW_SIZE, max_allocation=.5)
    valid_env = PortfolioEnv(valid_data,window_size=WINDOW_SIZE, max_allocation=.5)

    EPISODES = 50
    max_portfolio_manager = None
    max_reward = 0
    evaluate_every = 1
    logging.info(f'Starting training over {EPISODES} episodes, for {env.close_data.shape[1]} tickers, each with {env.close_data.shape[0]} steps, window size {WINDOW_SIZE}')

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        reward = train_porfolio_manager(env, trading_desk, episode)
        
        logging.info("Rendering validation environment...")
        valid_env.reset()
        reward_valid_dataset, steps, info = evaluate_steps_portfolio(valid_env,trading_desk, portfolio_manager)

        render_portfolio_summary(valid_env)
        
        logging.info(f"{info}, sigma noise: {portfolio_manager.noise.sigma}")
        evaluate_revards.append(info)
