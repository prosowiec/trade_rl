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
import matplotlib.pyplot as plt


from rl_agent_simple import DQNAgent
from rl_agent import load_dqn_agent
from source.database import read_stock_data
from copy import deepcopy
#from enviroments import TimeSeriesEnv_simple
from gym import spaces
from eval_portfolio import evaluate_steps_portfolio, render_env

import os

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"



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
            
        
        curr_value = self.cash + self.position * curr_price
        
        self.total_porfolio = curr_value
        portfolio_return = (curr_value - prev_value) / (prev_value + 1e-8) 
        reward = portfolio_return * 100 #np.log(1 + portfolio_return + 1e-8) - transaction_cost / (prev_value + 1e-8)

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


# class DQNPortfolio(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(DQNPortfolio, self).__init__()
#         self.out_steps = output_dim
#         self.lstm = nn.LSTM(input_size=input_dim, hidden_size=8, batch_first=True)
#         self.dropout = nn.Dropout(p=0.2)
#         self.fc = nn.Linear(8, output_dim)

#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)  # x: [batch, seq_len, features]
#         #last_hidden = lstm_out[:, -1, :]  # weź ostatni krok
#         #x = self.dropout(last_hidden)
#         x = self.fc(lstm_out)
#         x = x.view(-1, self.out_steps, 1)
#         x = torch.softmax(x, dim=1)  # alokacja portfela jako rozkład prawdopodobieństwa
#         return x

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
            nn.Sigmoid()  # Zakres [0, 1] dla każdej akcji
        )

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class AgentPortfolio:
    def __init__(self, input_dim=97, action_dim=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(input_dim, action_dim).to(self.device)
        self.target_actor = Actor(input_dim, action_dim).to(self.device)
        self.critic = Critic(input_dim, action_dim).to(self.device)
        self.target_critic = Critic(input_dim, action_dim).to(self.device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.loss_fn = nn.MSELoss()

        self.replay_memory = deque(maxlen=50000)
        self.MIN_REPLAY_MEMORY_SIZE = 300
        self.MINIBATCH_SIZE = 8
        self.DISCOUNT = 0.99
        self.TAU = 0.005  # do soft update

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state):
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        #print(actions)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        
        actions = np.array([action['portfolio_manager'] for action in actions])
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).view(-1, 1).to(self.device)

        # Krytyk - target Q
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards + (~dones) * self.DISCOUNT * target_q

        current_q = self.critic(states, actions)
        critic_loss = self.loss_fn(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Aktor - maksymalizacja oceny krytyka
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target sieci
        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)

    def _soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1.0 - self.TAU) * target_param.data)

    def get_action(self, state, noise_std=0.2):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        noise = np.random.normal(0, noise_std, size=action.shape)
        return np.clip(action + noise, 0, 1)
    
    def get_action_target(self, state):
        state = state.clone().detach().float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.target_actor(state).cpu().numpy()[0]

        return np.clip(action, 0, 1)


    



EPSILON_DECAY = 0.95

def train_episode(env, episode, epsilon):
    episode_reward = 0
    step = 1

    
    current_state = env.reset()
    done = False
    while not done:

        # if np.random.rand() < epsilon:
        #     action_allocation_percentages = torch.rand((1,), dtype=torch.float32)
        #     action_allocation_percentages = action_allocation_percentages.squeeze(0).cpu().numpy()  # [n_assets]
        # else:
        
        action_allocation_percentages = portfolio_manager.get_action(current_state)

        #action_allocation_percentages = torch.sigmoid(action_allocation_percentages)  # Ensure 0-1 range
        #action_allocation_percentages = action_allocation_percentages.squeeze(0).cpu().numpy()  # [n_assets]
        #print(action_allocation_percentages)

        action = {
            'trader': trader.get_action(current_state[:96], target_model = True),
            'portfolio_manager': np.array([action_allocation_percentages]).flatten()
        }
        
        new_state, reward, done, info = env.step(action)
        

        episode_reward += reward
        portfolio_manager.update_replay_memory((current_state, action, reward, new_state, done))
        
        #if np.random.random() >= .7:
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
portfolio_manager = AgentPortfolio()
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

max_portfolio_manager = None
max_reward = 0
evaluate_every = 1
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    reward = train_episode(env ,episode,epsilon)
    
    
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    
    #reward_all.append(reward)
    #if episode % evaluate_every:
    #render_env(valid_env)
    if not episode % 5:
        valid_env.reset()
        # if max_portfolio_manager:
        #     reward_valid_dataset, steps, info = evaluate_steps_portfolio(valid_env, trader, max_portfolio_manager)
        # else:
        reward_valid_dataset, steps, info = evaluate_steps_portfolio(valid_env, trader, portfolio_manager)

        render_env(valid_env)
    
    #print(env.portfolio_value_history)
        print(info)
        evaluate_revards.append(info)
    
    # if reward_valid_dataset > max_reward and episode > 1:
    #     max_reward = reward_valid_dataset
    #     #print(max_reward)
    #     max_portfolio_manager = deepcopy(portfolio_manager)
    
    # #nadpisz jeśli się pogorszy
    # if max_reward > 0 and episode > 1 and reward_valid_dataset / max_reward <= .7:
    #     agent = deepcopy(max_portfolio_manager)
