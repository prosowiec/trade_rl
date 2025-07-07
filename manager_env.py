import pandas as pd
import numpy as np
import gym
from gym import spaces
import ta 

class PortfolioEnv(gym.Env):
    def __init__(self, close_data: pd.Series, window_size=96, initial_cash=1.0,
                 transaction_cost=0.001, max_allocation=0.5):
        """
        close_data: Series z cenami jednego aktywa, indeks = daty
        """
        super(PortfolioEnv, self).__init__()
        self.close_data = close_data
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_allocation = max_allocation
        self.trader_action = 0
        self.total_porfolio = initial_cash
        self.cash = initial_cash
        self.action_space = spaces.Dict({
            'trader': spaces.Discrete(3),  # 0=hold, 1=buy, 2=sell
            'portfolio_manager': spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
        })

        # Obserwacja = okno cen + pozycja + gotówka + poprzednia akcja + alokacja
        obs_len = self.window_size + 1 + 1 + 1 + 1  # close + position + cash + prev_trader + prev_alloc
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_len,), dtype=np.float32)
        self.position = 0
        
        self.states_buy = []
        self.states_sell = []
        self.states_allocation = []
        self.shares_sell = []
        self.shares_buy = []
        self.states_position = []
        
        self.reset()

    def _get_obs(self):
        window_start = max(0, self.current_step - self.window_size)
        window = self.close_data.iloc[window_start:self.current_step].values.flatten()
        
        min_val = np.min(window)
        max_val = np.max(window)
        market_data = ((window - min_val) / (max_val - min_val + 1e-8)).astype(np.float32)
        
        mean_price = np.mean(market_data)
        std_price = np.std(market_data)

        current_price = self.close_data.iloc[self.current_step]
        total_value = self.cash + self.position * current_price
        
        # Normalizacje względem wartości początkowej
        norm_cash = self.cash / self.initial_cash
        norm_total_value = total_value / self.initial_cash
        norm_price = current_price / self.close_data.iloc[0]
        norm_position = (self.position * current_price) / self.initial_cash

        # Alokacje już są w [0, 1]
        #cash_alloc = self.cash / (total_value + 1e-8)
        #asset_alloc = (self.position * current_price) / (total_value + 1e-8)

        trader_action = self.trader_action
        # if self.trader_action == 2:
        #     trader_action = -1
        
        portfolio_info = np.array([
            #float(norm_cash),
            #float(norm_position),
            #float(norm_price),
            #float(norm_total_value),
            #float(cash_alloc),
            #float(asset_alloc),
            #float(mean_price),
            #float(std_price),
            float(self.trader_action)
        ], dtype=np.float32)
        #trader_info = np.array([self.trader_action / 2.0], dtype=np.float32)  # 0, 0.5, 1
        #return portfolio_info
        #return portfolio_info
        return np.round(np.concatenate([
                market_data,
                portfolio_info # Normalize trader action to [0, 0.5, 1]
            ]),4)        
        # current_price = self.close_data.iloc[self.current_step].values[0]

        # total_value = self.cash + self.position * current_price
        # asset_alloc = (self.position * current_price) / (total_value + 1e-8)
        # cash_alloc = self.cash / (total_value + 1e-8)

        
        # res = np.concatenate([
        #     market_data,
        #     [cash_alloc],
        #     [asset_alloc]
        #     , [self.trader_action]
        #     #cash_ratio,
        #     #position_ratio.values,
        #     #np.array(self.trader_action).flatten()
        #     # [position_ratio],
        #     # [cash_ratio],
        #     # [self.prev_trader_action / 2.0],
        #     # [self.prev_allocation]
        # ])
        # #print(res)
        # return res

    def reset(self):
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.position = 0.0  # liczba akcji
        self.prev_trader_action = 0
        self.prev_allocation = 0.0
        self.trader_action = 0
        self.portfolio_value_history = []
        self.total_porfolio = self.initial_cash
        
        self.shares_sell = []
        self.shares_buy = []
        self.states_buy = []
        self.states_sell = []
        self.states_allocation = []
        self.states_position = []
        self.cash_state = []

        return self._get_obs()


    def step(self, action):
        if self.current_step >= len(self.close_data):
            return self._get_obs(), 0.0, True, {}

        self.trader_action = action['trader']
        allocation = action['portfolio_manager'][0] #* self.max_allocation

        if float(allocation) == 0.0:
            # Return penalty and current state unchanged
            return self._get_obs(), 0, False, {'error': 'Invalid allocation'}
        
        if np.isnan(allocation) or allocation < 0 or allocation > 1:
            # Return penalty and current state unchanged
            return self._get_obs(), -1, False, {'error': 'Invalid allocation'}

        
        #print(allocation)
        curr_price = self.close_data.iloc[self.current_step].values[0]
        prev_price = self.close_data.iloc[self.current_step - 1].values[0]

        prev_value = self.cash + self.position * prev_price
        
        transaction_cost = 0.0
        executed = False
        reward = -1
        invest_amount = 0
        if self.trader_action == 1:  # BUY
            max_cash_available = self.cash
            max_invest = max_cash_available / (1 + self.transaction_cost)
            invest_amount = max_invest * allocation


            cost = invest_amount * self.transaction_cost
            total_spend = invest_amount + cost                
            
            if total_spend <= self.cash and invest_amount > 0:
                shares = invest_amount / curr_price
                self.position += shares
                self.cash -= total_spend
                transaction_cost += cost
                executed = True
                self.states_buy.append(self.current_step)
                self.shares_buy.append(shares)
             
                
                
        elif self.trader_action == 2:  # SELL
            shares = self.position * allocation  # ilość udziałów do sprzedaży

            if shares > 0:
                revenue = shares * curr_price
                cost = revenue * self.transaction_cost
                self.position -= shares
                self.cash += (revenue - cost)
                transaction_cost += cost
                executed = True
                self.states_sell.append(self.current_step)
                self.shares_sell.append(shares)
        
        if executed:
            self.states_allocation.append(allocation)
            self.states_position.append(self.position)
            
        self.cash_state.append(self.cash)
        curr_value = self.cash + self.position * curr_price
        
        self.total_porfolio = curr_value
        portfolio_return = (curr_value - prev_value) / (prev_value + 1e-8) 
        reward = portfolio_return   #np.log(1 + portfolio_return + 1e-8) - transaction_cost / (prev_value + 1e-8)
        
        
        # if self.trader_action != 0 and not executed:
        #     reward -= 200 # kara za niewykonaną transakcję
        
        # if invest_amount > self.cash and self.trader_action == 1:
        #     reward -= 10

        # elif executed:
        #     reward += 50

        self.prev_trader_action = self.trader_action
        self.prev_allocation = allocation
        self.portfolio_value_history.append(curr_value)
        self.current_step += 1

        done = self.current_step >= len(self.close_data) - 1
        info = {
            'portfolio_value': curr_value,
            'cash': self.cash,
            'position': self.position,
            'transaction_cost': transaction_cost,
            'return': portfolio_return,
            'executed': executed
        }
        reward = np.clip(reward, -1.0, 1.0)  
        
        return self._get_obs(), reward, done, info
        
    def get_portfolio_allocation(self):
        price = self.close_data.iloc[self.current_step - 1] if self.current_step > 0 else self.close_data.iloc[0]
        total_value = self.cash + self.position * price
        return {
            'asset_allocation': (self.position * price) / (total_value + 1e-8),
            'cash_allocation': self.cash / (total_value + 1e-8),
            'total_value': total_value,
            'shares': self.position
        }

    def sample_action(self):
        return {
            'trader': self.action_space['trader'].sample(),
            'portfolio_manager': self.action_space['portfolio_manager'].sample()
        }

    def get_price_window(self):
        window_start = max(0, self.current_step - self.window_size)
        window = self.close_data.iloc[window_start:self.current_step].values.flatten()
        min_val = np.min(window)
        max_val = np.max(window)
        market_data = ((window - min_val) / (max_val - min_val + 1e-8)).astype(np.float32)
        return market_data