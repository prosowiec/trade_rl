import pandas as pd
import numpy as np
import gym
from gym import spaces
import ta 

class PortfolioEnvTest(gym.Env):
    def __init__(self, close_data: pd.Series, window_size=96, initial_cash=50000.0,
                 transaction_cost=0.001, max_allocation=0.5):
        super(PortfolioEnvTest, self).__init__()
        self.close_data = close_data
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_allocation = max_allocation

        # Precompute technical indicators
        self.df = pd.DataFrame(close_data)
        self._add_technical_indicators()
        self.df.dropna(inplace=True)
        self.close_data = self.df[['close']]  # Nadpisz, by był zgodny dalej
        self.tech_indicators = self.df[self.indicator_cols].values

        self.action_space = spaces.Dict({
            'trader': spaces.Discrete(3),  # 0=hold, 1=buy, 2=sell
            'portfolio_manager': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })

        obs_len = len(self.indicator_cols) + 2  # wskaźniki + cash_alloc + asset_alloc
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)
        
        self.states_buy = []
        self.states_sell = []
        self.states_allocation = []


        self.reset()

    def _add_technical_indicators(self):
        df = self.df
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi() / 100.0
        df['macd'] = ta.trend.MACD(df['close']).macd() / 100.0
        df['ema_10'] = ta.trend.EMAIndicator(df['close'], window=10).ema_indicator() / df['close']
        df['ema_30'] = ta.trend.EMAIndicator(df['close'], window=30).ema_indicator() / df['close']
        df['stoch_k'] = ta.momentum.StochasticOscillator(df['close'], df['close'], df['close']).stoch() / 100.0
        df['cci'] = ta.trend.CCIIndicator(df['close'], df['close'], df['close'], window=20).cci() / 100.0
        df['adx'] = ta.trend.ADXIndicator(df['close'], df['close'], df['close']).adx() / 100.0
        df['willr'] = ta.momentum.WilliamsRIndicator(df['close'], df['close'], df['close']).williams_r() / -100.0
        df['roc'] = ta.momentum.ROCIndicator(df['close']).roc() / 100.0
        df['sma_ratio'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator() / df['close']
        self.indicator_cols = ['rsi', 'macd', 'ema_10', 'ema_30', 'stoch_k', 'cci', 'adx', 'willr', 'roc', 'sma_ratio']

    def _get_obs(self):
        idx = self.current_step
        if idx >= len(self.tech_indicators):
            idx = len(self.tech_indicators) - 1

        indicators = self.tech_indicators[idx]
        current_price = self.close_data.iloc[idx].values[0]

        total_value = self.cash + self.position * current_price
        asset_alloc = (self.position * current_price) / (total_value + 1e-8)
        cash_alloc = self.cash / (total_value + 1e-8)

        obs = np.concatenate([indicators, [cash_alloc, asset_alloc, self.trader_action]])
        return obs.astype(np.float32)
    
    def get_price_window(self):
        window_start = max(0, self.current_step - self.window_size)
        window = self.close_data.iloc[window_start:self.current_step].values.flatten()
        min_val = np.min(window)
        max_val = np.max(window)
        market_data = ((window - min_val) / (max_val - min_val + 1e-8)).astype(np.float32)
        return market_data
    
    def reset(self):
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.position = 0.0
        self.prev_trader_action = 0
        self.prev_allocation = 0.0
        self.trader_action = 0
        self.portfolio_value_history = []
        self.total_porfolio = self.initial_cash
        
        self.states_buy = []
        self.states_sell = []
        self.states_allocation = []

        return self._get_obs()

    def step(self, action):
        if self.current_step >= len(self.close_data):
            return self._get_obs(), 0.0, True, {}

        self.trader_action = action['trader']
        allocation = action['portfolio_manager'][0] #* self.max_allocation

        #print(allocation)
        curr_price = self.close_data.iloc[self.current_step].values[0]
        prev_price = self.close_data.iloc[self.current_step - 1].values[0]

        prev_value = self.cash + self.position * prev_price
        
        transaction_cost = 0.0
        executed = False
        
        if self.trader_action == 1:  # BUY
            # Maksymalna kwota, jaką można wydać z uwzględnieniem kosztu transakcyjnego
            # x + x * tc = cash → x = cash / (1 + tc)
            max_invest = self.cash / (1 + self.transaction_cost)
            
            invest_amount = max_invest * allocation  # allocation ∈ [0, 1]
            
            if invest_amount > 0:
                shares = invest_amount / curr_price
                cost = invest_amount * self.transaction_cost
                self.position += shares
                self.cash -= (invest_amount + cost)
                transaction_cost += cost
                executed = True
                self.states_buy.append(self.current_step)
                #self.shares_buy.append(shares)
        
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
                #self.shares_sell.append(shares)
        
        if executed:
            self.states_allocation.append(allocation)
            
        
        curr_value = self.cash + self.position * curr_price
        
        self.total_porfolio = curr_value
        portfolio_return = (curr_value - prev_value) / (prev_value + 1e-8) 
        reward = portfolio_return #np.log(1 + portfolio_return + 1e-8) - transaction_cost / (prev_value + 1e-8)

        if self.trader_action != 0 and not executed:
            reward -= 0.01  # kara za niewykonaną transakcję

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

        return self._get_obs(), reward, done, info

    def get_portfolio_allocation(self):
        price = self.close_data.iloc[self.current_step - 1].values[0]
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



class PortfolioEnv(gym.Env):
    def __init__(self, close_data: pd.Series, window_size=96, initial_cash=500000.0,
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
        self.action_space = spaces.Dict({
            'trader': spaces.Discrete(3),  # 0=hold, 1=buy, 2=sell
            'portfolio_manager': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })

        # Obserwacja = okno cen + pozycja + gotówka + poprzednia akcja + alokacja
        obs_len = self.window_size + 1 + 1 + 1 + 1  # close + position + cash + prev_trader + prev_alloc
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)
        
        self.states_buy = []
        self.states_sell = []
        self.states_allocation = []
        self.shares_sell = []
        self.shares_buy = []

        
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
        cash_alloc = self.cash / (total_value + 1e-8)
        asset_alloc = (self.position * current_price) / (total_value + 1e-8)

        portfolio_info = np.array([
            float(norm_cash),
            float(norm_position),
            float(norm_price),
            float(norm_total_value),
            float(cash_alloc),
            float(asset_alloc),
            float(mean_price),
            float(std_price)
        ], dtype=np.float32)
        #trader_info = np.array([self.trader_action / 2.0], dtype=np.float32)  # 0, 0.5, 1

        return np.concatenate([portfolio_info, [self.trader_action]])
        
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

        return self._get_obs()

    def step(self, action):
        if self.current_step >= len(self.close_data):
            return self._get_obs(), 0.0, True, {}

        self.trader_action = action['trader']
        allocation = action['portfolio_manager'][0] #* self.max_allocation

        #print(allocation)
        curr_price = self.close_data.iloc[self.current_step].values[0]
        prev_price = self.close_data.iloc[self.current_step - 1].values[0]

        prev_value = self.cash + self.position * prev_price
        
        transaction_cost = 0.0
        executed = False
        #print(self.trader_action)
        if self.trader_action == 1:  # BUY
            # Maksymalna kwota, jaką można wydać z uwzględnieniem kosztu transakcyjnego
            # x + x * tc = cash → x = cash / (1 + tc)
            max_invest = self.cash #/ (1 + self.transaction_cost)
            
            invest_amount = max_invest * allocation  # allocation ∈ [0, 1]
            
            if invest_amount > 0:
                shares = invest_amount / curr_price
                cost = 0 #invest_amount * self.transaction_cost
                self.position += shares
                self.cash -= (invest_amount + cost)
                transaction_cost += cost
                executed = True
                self.states_buy.append(self.current_step)
                self.shares_buy.append(shares)
        
        elif self.trader_action == 2:  # SELL
            shares = self.position * allocation  # ilość udziałów do sprzedaży

            if shares > 0:
                revenue = shares * curr_price
                cost = 0 #revenue * self.transaction_cost
                self.position -= shares
                self.cash += (revenue - cost)
                transaction_cost += cost
                executed = True
                self.states_sell.append(self.current_step)
                self.shares_sell.append(shares)
        
        if executed:
            self.states_allocation.append(allocation)
            
        
        curr_value = self.cash + self.position * curr_price
        
        self.total_porfolio = curr_value
        
        
        portfolio_return = (curr_value - prev_value) / (prev_value + 1e-8) 
        reward = portfolio_return 
        #print(reward)
        # if self.current_step < 200:
        #     reward = portfolio_return * 100 #np.log(1 + portfolio_return + 1e-8) - transaction_cost / (prev_value + 1e-8)
        # else:
        #     reward = (curr_value - prev_value) / np.std(self.portfolio_value_history)
        if self.trader_action != 0 and not executed:
            reward -= 1 #ara za niewykonaną transakcję

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