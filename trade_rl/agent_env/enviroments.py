import gym
from gym import spaces
import numpy as np


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
            profit = (price - bought_price) #/ bought_price 
            reward = np.clip(profit, -1, 1)
            self.total_profit += profit
            #reward = profit
            self.states_sell.append(self.current_step)

        self.current_step += 1

        if self.current_step >= len(self.data):
            done = True
            if len(self.inventory) > 0:
                self.total_profit += np.sum(self.data[-1] - np.array(self.inventory, dtype=np.float32))


        return self._get_observation(), reward, done
