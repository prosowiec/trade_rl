import gym
from gym import spaces
import numpy as np
import torch
import pandas as pd
import ta  

class TimeSeriesEnv(gym.Env):
    def __init__(self, data,min_val, max_val, window_size=10):
        super(TimeSeriesEnv, self).__init__()
        self.data = data
        self.window_size = window_size
        self.current_step = window_size

        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(window_size,), dtype=np.float32
        )

        self.inventory = []
        self.total_profit = 0.0
        self.states_buy = []
        self.states_sell = []
        
        self.min_val = min_val#np.min(data)
        self.max_val = max_val#np.max(data)

    def reset(self):
        self.current_step = self.window_size
        self.inventory = []
        self.total_profit = 0.0
        self.states_buy = []
        self.states_sell = []
        return self._get_observation()

    def _get_observation(self):
        past = self.data[self.current_step - self.window_size:self.current_step]
        #future = self.data[self.current_step:self.current_step + self.future_size]
        #obs = np.concatenate([past, future])

        # Normalizacja do zakresu [0, 1]
        return ((past - self.min_val) / (self.max_val - self.min_val + 1e-8)).astype(np.float32)

    def step(self, action):
        done = False
        reward = 0.0
        price = self.data[self.current_step]

        if action == 1:  # Buy
            self.inventory.append(price)
            self.states_buy.append(self.current_step)

        elif action == 2 and len(self.inventory) > 0:  # Sell
            bought_price = self.inventory.pop(0)
            profit = price - bought_price
            reward = profit #max(profit, 0)
            self.total_profit += profit
            #reward = self.total_profit
            self.states_sell.append(self.current_step)

        # Hold (0) nic nie robi

        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True

        return self._get_observation(), reward, done



class TimeSeriesEnvFuture(gym.Env):
    def __init__(self, data, window_size=10, future_size=48):
        self.data = data
        self.window_size = window_size
        self.future_size = future_size
        self.current_step = window_size

        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(window_size + future_size,), dtype=np.float32
        )

        self.inventory = []
        self.total_profit = 0.0
        self.states_buy = []
        self.states_sell = []

        self.min_val = np.min(data)
        self.max_val = np.max(data)

    def reset(self):
        self.current_step = self.window_size
        self.inventory = []
        self.total_profit = 0.0
        self.states_buy = []
        self.states_sell = []

        if self.current_step + self.future_size >= len(self.data):
            raise ValueError("Dane są zbyt krótkie, by zobaczyć przyszłe punkty.")

        return self._get_observation()

    def _get_observation(self):
        past = self.data[self.current_step - self.window_size:self.current_step]
        future = self.data[self.current_step:self.current_step + self.future_size]
        obs = np.concatenate([past, future])

        # Normalizacja do zakresu [0, 1]
        return ((obs - self.min_val) / (self.max_val - self.min_val + 1e-8)).astype(np.float32)

    def step(self, action):
        done = False
        reward = 0.0

        price = self.data[self.current_step]

        if action == 1:  # Buy
            self.inventory.append(price)
            self.states_buy.append(self.current_step)

        elif action == 2 and len(self.inventory) > 0:  # Sell
            bought_price = self.inventory.pop(0)
            profit = price - bought_price
            reward = profit
            self.total_profit += profit
            self.states_sell.append(self.current_step)

        # Hold (0) -> brak zmiany stanu

        self.current_step += 1
        if self.current_step + self.future_size >= len(self.data):
            done = True

        return self._get_observation(), reward, done


class TimeSeriesEnvFuturePredict(gym.Env):
    def __init__(self, data,lstm,lstm_data,train_std,train_mean,device, window_size=10, future_size=16) :
        self.data = data
        self.window_size = window_size
        self.future_size = future_size
        self.current_step = window_size
        self.device = device
        self.lstm = lstm
        self.lstm_data = lstm_data
        self.train_std = train_std
        self.train_mean = train_mean
        
        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = spaces.Box(
            low=-50, high=50, shape=(window_size + future_size,), dtype=np.float32
        )

        self.inventory = []
        self.total_profit = 0.0
        self.states_buy = []
        self.states_sell = []

        self.min_val = np.min(data)
        self.max_val = np.max(data)

    def reset(self):
        self.current_step = self.window_size
        self.inventory = []
        self.total_profit = 0.0
        self.states_buy = []
        self.states_sell = []

        if self.current_step >= len(self.data):
            raise ValueError("Dane są zbyt krótkie, by zobaczyć przyszłe punkty.")

        return self._get_observation()

    def _get_observation(self):
        past = self.data[self.current_step - self.window_size:self.current_step]
        past_lstm = self.lstm_data[self.current_step - self.window_size:self.current_step]
        #future = self.data[self.current_step:self.current_step + self.future_size]
        inputs = torch.from_numpy(past_lstm.astype(np.float32)).to(self.device)
        outputs  = self.lstm(inputs)
        future = outputs.detach().cpu().numpy().flatten()# * train_std + train_mean
        past = (past - self.train_mean) / self.train_std
        obs = np.concatenate([past,future])
        # (train_df - train_mean) / train_std
        #obs = (obs - self.train_mean) / self.train_std
        return np.round(obs.astype(np.float32),3)

    def step(self, action):
        done = False
        reward = 0.0
        price = self.data[self.current_step]

        if action == 1:  # Buy
            self.inventory.append(price)
            self.states_buy.append(self.current_step)
            # Brak nagrody za samo kupno

        elif action == 2 and len(self.inventory) > 0:  # Sell
            bought_price = self.inventory.pop(0)
            profit = price - bought_price
            reward = profit #max(profit, 0)  
            self.total_profit += profit
            #reward = self.total_profit
            self.states_sell.append(self.current_step)

        # Hold (0) nic nie robi

        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True
            self.total_profit = self.total_profit + np.sum(self.data[-1] - self.inventory ) 

        return self._get_observation(), reward, done

class TimeSeriesEnvFuturePredict_VEC(gym.Env):
    def __init__(self, data, lstm, lstm_data, train_std, train_mean, device, window_size=10, future_size=16):
        self.data = data.astype(np.float32)
        self.window_size = window_size
        self.future_size = future_size
        self.current_step = window_size
        self.device = device
        self.lstm = lstm
        self.lstm_data = lstm_data.astype(np.float32)
        self.train_std = float(train_std)
        self.train_mean = float(train_mean)

        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = spaces.Box(
            low=-50, high=50, shape=(window_size + future_size,), dtype=np.float32
        )

        self.inventory = []
        self.total_profit = 0.0
        self.states_buy = []
        self.states_sell = []

    def reset(self, *, seed=None, options=None):
        self.current_step = self.window_size
        self.inventory = []
        self.total_profit = 0.0
        self.states_buy = []
        self.states_sell = []

        if self.current_step >= len(self.data):
            raise ValueError("Dane są zbyt krótkie, by zobaczyć przyszłe punkty.")

        return self._get_observation(), {}

    def _get_observation(self):
        past = self.data[self.current_step - self.window_size:self.current_step]
        past_lstm = self.lstm_data[self.current_step - self.window_size:self.current_step].astype(np.float32)

        inputs = torch.from_numpy(past_lstm.astype(np.float32)).float().to(self.device)
        outputs = self.lstm(inputs)
        future = outputs.detach().cpu().numpy().flatten().astype(np.float32)

        past = (past - self.train_mean) / self.train_std
        obs = np.concatenate([past, future])
        return np.round(obs.astype(np.float32), 3)

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0.0
        price = self.data[self.current_step]

        if action == 1:  # Buy
            self.inventory.append(price)
            self.states_buy.append(self.current_step)

        elif action == 2 and len(self.inventory) > 0:  # Sell
            bought_price = self.inventory.pop(0)
            profit = price - bought_price
            reward = profit
            self.total_profit += profit
            self.states_sell.append(self.current_step)

        self.current_step += 1

        if self.current_step >= len(self.data):
            terminated = True
            if len(self.inventory) > 0:
                self.total_profit += np.sum(self.data[-1] - np.array(self.inventory, dtype=np.float32))

        return self._get_observation(), reward, terminated, truncated, {}

class TimeSeriesEnv_simple(gym.Env):
    def __init__(self, data, window_size=10):
        super(TimeSeriesEnv_simple, self).__init__()
        self.data = data
        self.window_size = window_size
        self.current_step = window_size

        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(window_size,), dtype=np.float32
        )

        self.inventory = []
        self.total_profit = 0.0
        self.states_buy = []
        self.states_sell = []
        self.action_reward = 0
        #self.max_size = 10
        
        self.min_val = np.min(data)
        self.max_val = np.max(data)

    def reset(self):
        self.current_step = self.window_size
        self.inventory = []
        self.total_profit = 0.0
        self.states_buy = []
        self.states_sell = []
        return self._get_observation()

    def _get_observation(self):
        past = self.data[self.current_step - self.window_size:self.current_step]

        self.min_val = np.min(past)
        self.max_val = np.max(past)
        return ((past - self.min_val) / (self.max_val - self.min_val + 1e-8)).astype(np.float32)

    def step(self, action):
        done = False
        reward = 0.0
        price = self.data[self.current_step]

        if action == 1:  # Buy
            self.inventory.append(price)
            self.states_buy.append(self.current_step)

        elif action == 2 and len(self.inventory) > 0:  # Sell
            bought_price = self.inventory.pop(0)
            profit = price - bought_price
            reward = profit 
            self.total_profit += profit
            #reward = self.total_profit
            self.states_sell.append(self.current_step)

        self.current_step += 1

        if self.current_step >= len(self.data):
            done = True
            if len(self.inventory) > 0:
                self.total_profit += np.sum(self.data[-1] - np.array(self.inventory, dtype=np.float32))


        return self._get_observation(), reward, done
    

class TimeSeriesEnvOHLC(gym.Env):
    def __init__(self, data, window_size=10):
        super(TimeSeriesEnvOHLC, self).__init__()
        self.window_size = window_size
        self.df = data.copy()

        # Wskaźniki techniczne
        self.df['SMA'] = ta.trend.sma_indicator(self.df['close'], window=14)
        self.df['EMA'] = ta.trend.ema_indicator(self.df['close'], window=14)
        self.df['RSI'] = ta.momentum.rsi(self.df['close'], window=14)
        self.df['MACD'] = ta.trend.macd_diff(self.df['close'])
        self.df['BB_upper'] = ta.volatility.bollinger_hband(self.df['close'])
        self.df['BB_lower'] = ta.volatility.bollinger_lband(self.df['close'])
        median_price = (self.df['high'] + self.df['low']) / 2
        self.df['MOM'] = median_price.rolling(window=5).mean() - median_price.rolling(window=34).mean()

        self.df.fillna(0.0, inplace=True)

        self.ohlc_data = self.df[['open', 'high', 'low', 'close',
                                  'SMA', 'EMA', 'RSI', 'MACD',
                                  'BB_upper', 'BB_lower', 'MOM']].values

        self.current_step = window_size
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(window_size, self.ohlc_data.shape[1]), dtype=np.float32)

        self.initial_cash = 100_000.0
        self.cash = self.initial_cash
        self.inventory = 0.0  # liczba akcji
        self.total_profit = 0.0
        self.allocations = []
        self.last_portfolio_value = self.initial_cash
        self.states_buy = []
        self.states_sell = []

    def reset(self):
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.inventory = 0.0
        self.total_profit = 0.0
        self.allocations = []
        self.last_portfolio_value = self.initial_cash
        self.states_buy = []
        self.states_sell = []

        return self._get_observation()

    def _get_observation(self):
        window = self.ohlc_data[self.current_step - self.window_size:self.current_step]
        min_val = np.min(window, axis=0)
        max_val = np.max(window, axis=0)
        norm_window = (window - min_val) / (max_val - min_val + 1e-8)

        price = self.ohlc_data[self.current_step - 1][3]
        position_value = self.inventory * price
        portfolio_value = self.cash + position_value
        position_ratio = position_value / portfolio_value if portfolio_value > 0 else 0.0
        last_action = self.allocations[-1] if self.allocations else 0
        return (norm_window.astype(np.float32).T, [last_action])

    def step(self, action):
        done = False
        action = float(action[0])
        confidence = abs(action)
        allocation = min(confidence, 1.0)
        price = self.ohlc_data[self.current_step][3]
        last_allocation = self.allocations[-1] if self.allocations else 0.0
       # action = action - self.allocations[-1] if self.allocations else 0  # Adjust action based on last allocation
        
        BUY_SELL = action - last_allocation
        # Buy
        #print(action)
        if  action > 0:
            #to_buy = action - last_allocation
            invest_amount = self.cash * allocation
            
            #invest_amount = self.cash * to_buy
            invest_amount = min(invest_amount, self.cash)
            shares = invest_amount / price
            self.inventory += shares
            self.cash -= invest_amount
            self.states_buy.append(self.current_step)


        # Sell
        elif  action < 0  and self.inventory > 0:
            shares_to_sell = self.inventory * allocation
            #to_sell = last_allocation - action
            #shares_to_sell = self.inventory * abs(to_sell)
            shares_to_sell = min(self.inventory, shares_to_sell)
            revenue = shares_to_sell * price
            self.inventory -= shares_to_sell
            self.cash += revenue
            self.states_sell.append(self.current_step)


        # Reward — zwrot portfela + entropia
        curr_portfolio_value = self.cash + self.inventory * price
        portfolio_return = (curr_portfolio_value - self.last_portfolio_value) / (self.last_portfolio_value + 1e-8)# * allocation
        self.last_portfolio_value = curr_portfolio_value

        entropy_coeff = 0.01
        entropy = -(
            allocation * np.log(allocation + 1e-8) +
            (1 - allocation) * np.log(1 - allocation + 1e-8)
        )
        reward = portfolio_return * 100 #+ entropy_coeff * entropy
        reward = np.clip(reward, -1.0, 1.0)

        self.current_step += 1
        self.allocations.append(action)

        if self.current_step >= len(self.ohlc_data):
            done = True
            final_price = self.ohlc_data[self.current_step - 1][3]
            self.cash += self.inventory * final_price
            self.inventory = 0.0
            self.total_profit = self.cash - self.initial_cash

        print(f"Step: {self.current_step} | Action: {action:3.2f} | Reward: {reward:4.4f} | Cash: {self.cash:6.2f} | Inv: {self.inventory:6.2f} | Value: {curr_portfolio_value:6.2f}")

        return self._get_observation(), reward, done