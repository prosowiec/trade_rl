import pandas as pd
import numpy as np
import gym
from gym import spaces
from managerReward import DifferentialSharpeRatio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # możesz dodać np. FileHandler
)


class PortfolioEnv(gym.Env):
    def __init__(self, close_data: pd.DataFrame, window_size=96, initial_cash=10000.0,
                 transaction_cost=1, max_allocation=0.3, show_step_info = False):
        """
        close_data: DataFrame z cenami wielu aktywów (każda kolumna to inne aktywo)
        """
        super(PortfolioEnv, self).__init__()
        self.close_data = close_data
        self.asset_names = close_data.columns.tolist()
        self.n_assets = len(self.asset_names)
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_allocation = max_allocation
        self.total_portfolio = initial_cash
        
        self.show_step_info = show_step_info
        self.dsr = DifferentialSharpeRatio(eta=0.01)

        self.action_space = spaces.Dict({
            'trader': spaces.MultiDiscrete([3] * self.n_assets),  # 0=hold, 1=buy, 2=sell per asset
            'portfolio_manager': spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        })

        obs_len = self.n_assets * window_size + self.n_assets + 1 + self.n_assets + self.n_assets
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_len,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.total_portfolio = self.initial_cash
        self.position = np.zeros(self.n_assets)  # ilość akcji każdego aktywa
        self.prev_trader_action = np.zeros(self.n_assets)
        self.prev_allocation = np.zeros(self.n_assets)
        self.trader_action = np.zeros(self.n_assets)
        self.portfolio_value_history = []
        
        self.states_buy = [[] for _ in range(self.n_assets)]
        self.states_sell = [[] for _ in range(self.n_assets)]
        self.states_allocation = [[] for _ in range(self.n_assets)]
        self.shares_buy = [[] for _ in range(self.n_assets)]
        self.shares_sell = [[] for _ in range(self.n_assets)]
        self.asset_value_history = [[] for _ in range(self.n_assets)]
        
        self.asset_percentage_sell_history = [[] for _ in range(self.n_assets)]
        self.asset_percentage_buy_history = [[] for _ in range(self.n_assets)]
        
        return self._get_obs()

    def _get_obs(self):
        window_start = max(0, self.current_step - self.window_size)
        curr_prices = self.close_data.iloc[window_start:self.current_step].values  # shape: (window, n_assets)

        min_vals = self.close_data.min().values
        max_vals = self.close_data.max().values
        norm_prices = (curr_prices - min_vals) / (max_vals - min_vals + 1e-8)  # shape: (n_assets,)


        norm_actions = self.trader_action #/ 2.0
        for i in range(self.n_assets):
            if self.prev_trader_action[i] == 1:
                norm_actions[i] = 1.0 
            elif self.prev_trader_action[i] == 2:
                norm_actions[i] = -1.0
            else:
                norm_actions[i] = 0.0
        norm_actions = norm_actions[:, np.newaxis]
        
        last_prices = self.close_data.iloc[self.current_step - 1].values
        asset_values = self.position * last_prices
        total_value = self.cash + np.sum(asset_values)
        portfolio_shares = asset_values / (total_value + 1e-8)   # udział każdego aktywa
        
        cash_share = self.cash / (total_value + 1e-8)            # udział gotówki

        portfolio_shares = portfolio_shares[:, np.newaxis]
        cash_share = cash_share * np.ones((self.n_assets, 1))  # cash share as a column vector

        obs = np.concatenate([norm_prices.T, norm_actions, portfolio_shares, cash_share], axis=1)  # shape: (n_assets, 2)
    
        return np.round(obs, 4)
    
    
    def step(self, action):
        executed = False
        self.trader_action = np.array(action['trader'])
        allocation = np.array(action['portfolio_manager'])
        curr_prices = self.close_data.iloc[self.current_step].values
        prev_prices = self.close_data.iloc[self.current_step - 1].values
        prev_value = self.cash + np.sum(self.position * prev_prices)

        
        for i in range(self.n_assets):
            act = self.trader_action[i]
            alloc = allocation[i]
            price = curr_prices[i]
            self.states_allocation[i].append(alloc)
            
            if act == 1:  # BUY

                current_allocation = (self.position[i] * price) / prev_value
                allocation_left = max(0, self.max_allocation - current_allocation)

                invest_amount = self.cash * allocation_left
                invest_amount = min(invest_amount + self.transaction_cost, self.cash)
                shares = np.floor(invest_amount / price)
                
                max_shares_to_buy = max(np.floor((self.cash - self.transaction_cost) / price),0)
                shares = min(shares, max_shares_to_buy)
                cost = shares * price + self.transaction_cost

                if shares > 0:
                    invest_amount = shares * price
                    self.position[i] += shares
                    self.cash -= cost
                    executed = True

                    logging.info(
                        f"Buying {shares} of asset {self.asset_names[i]} at price {price} "
                        f"with invest amount {invest_amount} and cost {cost}, cash now {self.cash}"
                    )
              
                self.states_buy[i].append(self.current_step)
                self.shares_buy[i].append(shares)
                self.asset_percentage_buy_history[i].append(shares * price / (prev_value + 1e-8))
                    
            elif act == 2:  # SELL
                shares_to_sell = self.position[i] * alloc
                               
                shares_to_sell = np.floor(min(shares_to_sell, self.position[i]))  # nie można sprzedać więcej niż się posiada
                if shares_to_sell > 0:
                    revenue = shares_to_sell * price
                    self.position[i] -= shares_to_sell
                    self.cash += revenue
                    executed = True

                    logging.info(
                        f"Selling {shares_to_sell} of asset {self.asset_names[i]} at price {price} "
                        f"with revenue {revenue}, cash now {self.cash}"
                    )     
                self.states_sell[i].append(self.current_step)
                self.shares_sell[i].append(shares_to_sell)
                self.asset_percentage_sell_history[i].append(shares_to_sell * price / (prev_value + 1e-8))
                
            else:
                self.shares_buy[i].append(0)
                self.shares_sell[i].append(0)
                self.asset_percentage_sell_history[i].append(0)
                self.asset_percentage_buy_history[i].append(0)

            
            asset_value = self.position[i] * curr_prices[i]
            self.asset_value_history[i].append(asset_value)



        curr_value = self.cash + np.sum(self.position * curr_prices)
        self.total_portfolio = curr_value
        
        portfolio_return = (curr_value - prev_value) / (prev_value + 1e-8)        
   
        reward = self.dsr.update(portfolio_return) * 100
        
        self.prev_trader_action = self.trader_action.copy()
        self.prev_allocation = allocation.copy()
        self.portfolio_value_history.append(curr_value)
        self.current_step += 1
        done = self.current_step >= len(self.close_data) - 1
        
        
        if self.show_step_info == True:
            logging.info(
                f"Step {self.current_step:<6d} | "
                f"Reward: {reward:>10.6f} | "
                f"Cash: {self.cash:>12.2f} | "
                f"Total val: {curr_value:>12.2f}"
            )
        info = {
            'portfolio_value': curr_value,
            'cash': self.cash,
            'position': self.position.copy(),
            'return': portfolio_return,
            'executed': executed
        }
        
        return self._get_obs(), reward, done, info


    def get_price_window(self):
        """
        Used for trader agent to see price history.
        """
        window_start = max(0, self.current_step - self.window_size)
        curr_prices = self.close_data.iloc[window_start:self.current_step].values  # shape: (window, n_assets)
        min_vals = self.close_data.min().values
        max_vals = self.close_data.max().values

        norm_prices = (curr_prices - min_vals) / (max_vals - min_vals + 1e-8)  # shape: (n_assets,)

        return np.array(norm_prices).T
    
