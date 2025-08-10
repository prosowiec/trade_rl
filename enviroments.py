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
        self.df['SMA'] = ta.trend.sma_indicator(self.df['close'], window=14)
        self.df['EMA'] = ta.trend.ema_indicator(self.df['close'], window=14)
        self.df['RSI'] = ta.momentum.rsi(self.df['close'], window=14)
        self.df['MACD'] = ta.trend.macd_diff(self.df['close'])
        self.df['BB_upper'] = ta.volatility.bollinger_hband(self.df['close'])
        self.df['BB_lower'] = ta.volatility.bollinger_lband(self.df['close'])
        median_price = (self.df['high'] + self.df['low']) / 2
        self.df['MOM'] = median_price.rolling(window=5).mean() - median_price.rolling(window=34).mean()
        
        self.df.fillna(0.0, inplace=True)  # brakujące wartości

        # OHLC + 7 wskaźników = 11 wymiarów
        self.indicator_data = self.df[['open', 'high', 'low', 'close',
                                       'SMA', 'EMA', 'RSI', 'MACD',
                                       'BB_upper', 'BB_lower', 'MOM']].values

        self.ohlc_data = self.indicator_data

        self.current_step = window_size

        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(window_size * 4,), dtype=np.float32
        )

        self.inventory = []
        self.total_profit = 0.0
        self.prev_profit = 0.0
        self.states_buy = []
        self.states_sell = []
        self.allocations = []
        self.allocations_stack = []
        self.min_val = np.min(self.ohlc_data)
        self.max_val = np.max(self.ohlc_data)

    def reset(self):
        self.current_step = self.window_size
        self.inventory = []
        self.total_profit = 0.0
        self.prev_profit = 0.0
        self.states_buy = []
        self.states_sell = []
        self.allocations = []
        self.allocations_stack = []
        self.last_price = self.ohlc_data[self.current_step - 1][3]  
        return self._get_observation()

    def _get_observation(self):
        window = self.ohlc_data[self.current_step - self.window_size:self.current_step]
        # Normalizacja okna
        min_val = np.min(window, axis=0)  # min per column
        max_val = np.max(window, axis=0)  # max per column
        
        #print(f"Min: {min_val}, Max: {max_val}")
        norm_window = (window - min_val) / (max_val - min_val + 1e-8)
        #price = window[-1][3]
        
        #total_position_value = sum(self.inventory) if self.inventory else 0.0
        position_ratio = len(self.inventory) / 100 #portfolio_value if portfolio_value > 0 else 0.0
        position_ratio = min(max(position_ratio, 0.0), 1.0)
        position_ratio = 0.5
        
        
        return (norm_window.astype(np.float32).T, [position_ratio])

    def step(self, action):
        done = False
        price = self.ohlc_data[self.current_step][3]
        last_price = self.ohlc_data[self.current_step - 1][3]
        action = action[0]

        reward = 0

        # BUY
        if action == 1 and len(self.inventory) <= 100:
            self.inventory.append(price)
            self.states_buy.append(self.current_step)
            self.allocations_stack.append(action)
            reward = 0

        # SELL
        elif action == 2 and len(self.inventory) > 0:
            bought_price = self.inventory.pop(0)
            bought_alocation = self.allocations_stack.pop(0)
            profit = price - bought_price
            self.total_profit += profit
            self.states_sell.append(self.current_step)

        
            confidence = (abs(action) + bought_alocation)
            reward = (profit ) #/ bought_price) * confidence * 100 + self.total_profit * 0.001

        self.prev_profit = self.total_profit
        self.current_step += 1
                

        print(f"Step: {self.current_step:>6.0f}  |Action: {action:>6.2f}  |  Reward: {reward:>6.2f} ")
        self.allocations.append(action)
        if self.current_step >= len(self.ohlc_data):
            done = True
            self.total_profit += np.sum(self.ohlc_data[-1][3] - np.array(self.inventory, dtype=np.float32))

        return self._get_observation(), reward, done
