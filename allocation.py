import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from collections import deque
import time
import random
from tqdm import tqdm
import os
import pandas as pd

from rl_agent_simple import DQNAgent
from rl_agent import load_dqn_agent
from source.database import read_stock_data
from copy import deepcopy
#from enviroments import TimeSeriesEnv_simple
from gym import spaces


import os

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"



class PortfolioEnv(gym.Env):
    def __init__(self, close_data: pd.Series, window_size=96, initial_cash=100000.0,
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

        self.reset()

    def _get_obs(self):
        window_start = max(0, self.current_step - self.window_size)
        window = self.close_data.iloc[window_start:self.current_step].values.flatten()
        
        min_val = np.min(window)
        max_val = np.max(window)
        market_data = ((window - min_val) / (max_val - min_val + 1e-8)).astype(np.float32)

        # if len(window) > 0:
        #     normalized_window = window / window.iloc[0]
        #     if len(normalized_window) < self.window_size:
        #         padding = np.zeros(self.window_size - len(normalized_window))
        #         market_data = np.concatenate([padding, normalized_window.values])
        #     else:
        #         market_data = normalized_window.values
        # else:
        #     market_data = np.zeros(self.window_size)

        current_price = self.close_data.iloc[self.current_step - 1] if self.current_step > 0 else self.close_data.iloc[0]
        total_value = self.cash + self.position * current_price
        
        #position_ratio = (self.position * current_price) / (total_value + 1e-8)
        #cash_ratio = self.cash / (total_value + 1e-8)
        # print(market_data.shape)
        # print(cash_ratio.values)
        # print(position_ratio.values)
        # print(self.trader_action)
        
        res = np.concatenate([
            market_data,
            [self.trader_action]
            #cash_ratio,
            #position_ratio.values,
            #np.array(self.trader_action).flatten()
            # [position_ratio],
            # [cash_ratio],
            # [self.prev_trader_action / 2.0],
            # [self.prev_allocation]
        ])
        #print(res)
        return res

    def reset(self):
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.position = 0.0  # liczba akcji
        self.prev_trader_action = 0
        self.prev_allocation = 0.0
        self.trader_action = 0
        self.portfolio_value_history = []
        self.total_porfolio = self.initial_cash
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
        
        elif self.trader_action == 2:  # SELL
            shares = self.position * allocation  # ilość udziałów do sprzedaży

            if shares > 0:
                revenue = shares * curr_price
                cost = revenue * self.transaction_cost
                self.position -= shares
                self.cash += (revenue - cost)
                transaction_cost += cost
                executed = True
        
        curr_value = self.cash + self.position * curr_price
        
        self.total_porfolio = curr_value
        portfolio_return = (curr_value - prev_value) # / (prev_value + 1e-8)
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


class DQNPortfolio(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNPortfolio, self).__init__()
        self.out_steps = output_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=8, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(8, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # x: [batch, seq_len, features]
        #last_hidden = lstm_out[:, -1, :]  # weź ostatni krok
        #x = self.dropout(last_hidden)
        x = self.fc(lstm_out)
        x = x.view(-1, self.out_steps, 1)
        x = torch.softmax(x, dim=1)  # alokacja portfela jako rozkład prawdopodobieństwa
        return x


class Agent_portfolio:
    def __init__(self, input_dim=96 + 1, action_dim=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.action_dim = action_dim

        self.model = DQNPortfolio(input_dim, action_dim).to(self.device)
        self.target_model = DQNPortfolio(input_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-2)
        self.loss_fn = nn.MSELoss()

        self.replay_memory = deque(maxlen=50000)
        self.MIN_REPLAY_MEMORY_SIZE = 300
        self.UPDATE_TARGET_EVERY = 1
        self.MINIBATCH_SIZE = 16
        self.DISCOUNT = 0.99
        self.target_update_counter = 0
        

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state):
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        actions = np.array([action['portfolio_manager'] for action in actions])
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).view(-1, 1)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device).view(-1, 1)

        with torch.no_grad():
            target_qs = self.target_model(next_states)
            max_future_qs = torch.max(target_qs, dim=1)[0].view(-1, 1)
            target = rewards + (~dones) * self.DISCOUNT * max_future_qs

        current_qs = self.model(states)  # [B, num_actions]
        predicted = current_qs.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B, 1]
        target = target.squeeze(1).view(-1, 1)
        print(actions)
        print(predicted.shape,target.shape )
        loss = self.loss_fn(predicted, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if terminal_state:
            self.target_update_counter += 1
        if self.target_update_counter >= self.UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, seq, features]
        with torch.no_grad():
            action = self.model(state)
        return action
    
def evaluate_steps_portfolio(env, trader, portfolio_manager, device="cuda:0"):
    """
    Evaluate the portfolio environment with separate trader and portfolio manager models
    
    Args:
        env: PortfolioEnv environment
        trader_model: Model that outputs trading decisions (buy/sell/hold)
        portfolio_model: Model that outputs allocation percentages
        device: Device to run models on
    """
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    allocations = []
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # [1, obs_size]
        trader_actions = trader.get_action(state[:96], target_model = True)
        with torch.no_grad():
            portfolio_allocations = portfolio_manager.target_model(state_tensor)  # [1, n_assets]
            portfolio_allocations = torch.sigmoid(portfolio_allocations)  # Ensure 0-1 range
            portfolio_allocations = portfolio_allocations.squeeze(0).cpu().numpy()  # [n_assets]
        
        # Combine actions
        action = {
            'trader': trader_actions,
            'portfolio_manager': portfolio_allocations
        }
        allocations.append(portfolio_allocations)
        # Take step
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Optional: print progress for debugging
        #if steps % 100 == 0:
        #    print(f"Step {steps}, Reward: {reward}, Portfolio Value: {info['portfolio_value']}")
    #print(allocations)
    return total_reward, steps, info['portfolio_value']



EPSILON_DECAY = 0.99

def train_episode(env, episode, epsilon):
    episode_reward = 0
    step = 1

    
    current_state = env.reset()
    done = False
    while not done:

        if np.random.rand() < epsilon:
            action_allocation_percentages = torch.rand((1,), dtype=torch.float32)
        else:
            action_allocation_percentages = portfolio_manager.get_action(current_state)

        #action_allocation_percentages = torch.sigmoid(action_allocation_percentages)  # Ensure 0-1 range
        action_allocation_percentages = action_allocation_percentages.squeeze(0).cpu().numpy()  # [n_assets]
        #print(action_allocation_percentages)

        action = {
            'trader': trader.get_action(current_state[:96], target_model = True),
            'portfolio_manager': np.array([action_allocation_percentages]).flatten()
        }
        
        new_state, reward, done, info = env.step(action)
        

        episode_reward += reward
        portfolio_manager.update_replay_memory((current_state, action, reward, new_state, done))
        
        if np.random.random() >= .7:
            portfolio_manager.train(done)

        current_state = new_state
        
        step += 1
 
    if not episode % 2:
            print(f"Episode: {episode} Total Reward: {env.total_porfolio} Epsilon: {epsilon:.2f}")

    return episode_reward



ticker = 'AAPL'
train_df, val_df ,rl_df,test_df = read_stock_data(ticker)
training_set = pd.concat([train_df, val_df ,rl_df,test_df])
training_set

trader = DQNAgent()
load_dqn_agent(trader, 'aapl_best_agent_vc_dimOPT.pth')


reward_all = []
evaluate_revards = []
portfolio_manager = Agent_portfolio()
epsilon = 1


data = training_set['close'].copy()
data[ticker] = training_set['close']
data = pd.DataFrame(data[ticker])


data_split = int(len(data)  * 0.8)

train_data = data[:data_split]
valid_data = data[data_split:]

WINDOW_SIZE = 96
env = PortfolioEnv(train_data, window_size=WINDOW_SIZE)
valid_env = PortfolioEnv(valid_data,window_size=WINDOW_SIZE)


#super dla 200, batch64
EPISODES = 50
MIN_EPSILON = 0.001

max_reward = 0
evaluate_every = 1
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    reward = train_episode(env ,episode,epsilon)
    
    
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    
    #reward_all.append(reward)
    #if episode % evaluate_every:
    valid_env.reset()
    reward_valid_dataset, steps, info = evaluate_steps_portfolio(valid_env, trader, portfolio_manager)
    #print(env.portfolio_value_history)
    print(info)
    evaluate_revards.append(info)
    
    if reward_valid_dataset > max_reward and episode > 10:
        max_reward = reward_valid_dataset
        #print(max_reward)
        max_agent = deepcopy(portfolio_manager)
    
    #nadpisz jeśli się pogorszy
    if max_reward > 0 and episode > 10 and reward_valid_dataset / max_reward <= .7:
        agent = deepcopy(max_agent)
