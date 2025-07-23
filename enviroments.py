import gym
from gym import spaces
import numpy as np
import torch

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
    
import gym
from gym import spaces
import numpy as np

class TimeSeriesEnvOHLC(gym.Env):
    def __init__(self, data, window_size=10):
        super(TimeSeriesEnvOHLC, self).__init__()
        self.window_size = window_size
        # self.grain = grain
        
        # # Przekształcamy dane na OHLC
        self.ohlc_data = data.values  # ucinamy do pełnych okien
        # data_reshaped = data.reshape(-1, grain)
        # self.ohlc_data = np.column_stack([
        #     data_reshaped[:, 0],                         # open
        #     np.max(data_reshaped, axis=1),               # high
        #     np.min(data_reshaped, axis=1),               # low
        #     data_reshaped[:, -1]                         # close
        # ])

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
        self.last_price = self.ohlc_data[self.current_step - 1][3]  
        return self._get_observation()

    def _get_observation(self):
        window = self.ohlc_data[self.current_step - self.window_size:self.current_step]
        # Normalizacja okna
        min_val = np.min(window)
        max_val = np.max(window)
        norm_window = (window - min_val) / (max_val - min_val + 1e-8)
        return norm_window.astype(np.float32).T

    # def step(self, action):
    #     done = False
    #     reward = -0.1
    #     price = self.ohlc_data[self.current_step][3]  # używamy 'close'
    #     self.last_price = self.ohlc_data[self.current_step - 1][3]  
    #     action = action[0]

    #     if action > 0.2:  # Buy
    #         self.inventory.append(price)
    #         #print(price)
    #         self.states_buy.append(self.current_step)
    #         #reward = price - self.last_price
    #         bought_price = self.inventory[0]
    #         self.allocations.append(action)
    #         reward += .1

    #     elif action < -0.2 and len(self.inventory) > 0:  # Sell
    #         bought_price = self.inventory.pop(0)
    #         prev_allocation = self.allocations.pop(0)   
    #         profit = (price - bought_price) / bought_price + 1
    #         self.total_profit += profit
    #         #reward = profit
            
    #         self.states_sell.append(self.current_step)
    #         #reward = np.clip(price - bought_price, -1.0, 1.0) #* (abs(action) + prev_allocation) / 2
    #         #profit = (price - bought_price) #* prev_allocation
    #         reward = reward * (abs(action) + prev_allocation) / 2
            
        
    #     print(reward, action)
    #     #reward -= 0.01 * len(self.inventory)  
    #     # if not self.inventory and action < 0.0:
    #     #     reward = -1
    #     # prev_price = self.ohlc_data[self.current_step - 1][3]
    #     # prev_value = sum([prev_price for _ in self.inventory])
        
    #     # current_value = sum([price for price in self.inventory])
    #     #reward = (current_value + self.total_profit) - (prev_value + self.prev_profit)
        
    #     self.prev_profit = self.total_profit
        
    #     self.current_step += 1

    #     if self.current_step >= len(self.ohlc_data):
    #         done = True
    #         if len(self.inventory) > 0:
    #             self.total_profit += np.sum(self.ohlc_data[-1][3] - np.array(self.inventory, dtype=np.float32))

    #     return self._get_observation(), reward, done

    def step(self, action):
        action = float(np.clip(action[0], -1, 1))  # zakres [-1, 1]

        done = False
        current_price = self.ohlc_data[self.current_step][3]
        
        if action > 0.2:  # Buy
            self.inventory.append(current_price)
            self.states_buy.append(self.current_step)
            self.allocations.append(abs(action))
            reward = -0.01  # Small penalty for buying to discourage overtrading
        elif action < -0.2 and len(self.inventory) > 0:  # Sell
            bought_price = self.inventory.pop(0)
            prev_allocation = self.allocations.pop(0)
            profit = current_price - bought_price
            reward = profit  # Reward based on realized profit
            self.total_profit += profit
            self.states_sell.append(self.current_step)
        else:  # Hold
            reward = -0.005 * len(self.inventory)  # Small penalty for holding
        
        
        #reward = profit  # <- kluczowa zmiana

        self.allocations.append(action)
            
        if self.current_step >= len(self.ohlc_data):
            done = True

        return self._get_observation(), reward, done