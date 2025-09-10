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
import copy

from trader import DQNAgent
from source.database import read_stock_data
from copy import deepcopy
#from enviroments import TimeSeriesEnv_simple
from gym import spaces
from eval_portfolio import evaluate_steps_portfolio, render_env, render_portfolio_summary
from manager_env import PortfolioEnv
import os


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):  # state_dim = [n_assets, features] = [4, 97]
        super(Actor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=96, out_channels=32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 2, kernel_size=1),
            nn.ReLU()
            
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                     # [B, 64, 4] → [B, 64*4]
            nn.Linear(2 * action_dim + 2 * action_dim, action_dim),
            #nn.Softmax(dim=1)
            #nn.Linear(8, action_dim),        # action_dim = liczba alokacji
            nn.Softmax(dim=-1)
            #nn.Softmax(dim=-1)                # ładne rozkłady alokacji
        )

    def forward(self, state):
        #print(state.shape)
        x = state.permute(0, 2, 1)  # [B, 4, 97] → [B, 97, 4]
        trader_actions = x[:, 96, :].unsqueeze(1)   # [B, 1, A]
        portfolio_features = x[:, 97, :].unsqueeze(1)      # [B, F-97, A] (jeśli istnieją)        x = x[:,:-1,:]
        x = x[:, :96, :]          # [B, 96, A]

        x = self.conv(x)            # [B, 64, 4]
        #print(f"x shape after conv: {x.shape}, trader_actions shape: {trader_actions.shape}, portfolio_features shape: {portfolio_features.shape}")
        x = torch.cat([x, trader_actions, portfolio_features], dim=1)
        x = self.fc(x)              # [B, 4]
        return x
        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=96, out_channels=32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 2, kernel_size=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                            # [B, 64, 4] → [B, 256]
            nn.Linear(2 * action_dim + 3 * action_dim , 8),      # dodajemy akcje
            #nn.ReLU(),
            # nn.Linear(8, action_dim)
        )

    def forward(self, state, action):
        x = state.permute(0, 2, 1)                   # [B, 97, 4]
        trader_actions = x[:, 96, :].unsqueeze(1)   # [B, 1, A]
        portfolio_features = x[:, 97, :].unsqueeze(1)      # [B, F-97, A] (jeśli istnieją)        x = x[:,:-1,:]
        x = x[:, :96, :]          # [B, 96, A]
        action = action.unsqueeze(1)  # [B, 1, 4]
        x = self.conv(x)                             # [B, 64, 4]
        x = torch.cat([x, action, trader_actions, portfolio_features], dim=1)            # [B, 256 + 4]
        
        x = self.fc(x)                               # [B, 1]
        return x
    
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.015, sigma=0.15):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_decay = 0.99995
        self.min_sigma = 0.01
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        if self.sigma > self.min_sigma:
            self.sigma *= self.sigma_decay
        #print(f"Current sigma: {self.sigma:.4f}")
        #self.state = (self.state - self.state.min()) / self.state.max()
        return self.state # np.random.normal(0, self.sigma, size=(1,self.size)) #
    
    def __call__(self, action):
        """Call to sample noise."""
        #print(action.flatten())
        res = action.flatten() + self.sample()
        min_val = np.min(res)
        max_val = np.max(res)

        #print(min_val, max_val, res)
        #res = (res - min_val) / (max_val - min_val)
        #print(res)
        #return res
        return np.clip(res,0,1)


class AgentPortfolio:
    def __init__(self, input_dim=96, action_dim=6): #27 * 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(input_dim, action_dim).to(self.device)
        self.target_actor = Actor(input_dim, action_dim).to(self.device)
        self.critic = Critic(input_dim, action_dim).to(self.device)
        self.target_critic = Critic(input_dim, action_dim).to(self.device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4, weight_decay=1e-2)

        self.loss_fn = nn.MSELoss()

        self.replay_memory = deque(maxlen=5000000)
        self.MIN_REPLAY_MEMORY_SIZE = 1000
        self.MINIBATCH_SIZE = 128
        self.DISCOUNT = 0.9995
        self.TAU = 1e-4  # do soft update

        self.noise = OUNoise(size=action_dim, mu=0.0, theta=0.15, sigma=0.5)

        
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state):
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        #print(actions)
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)

        actions = np.array([action['portfolio_manager'] for action in actions])
        actions = torch.from_numpy(np.array(actions, dtype=np.float32)).to(self.device)
        #print(rewards)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).to(self.device)

        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        #dones = dones.unsqueeze(1).expand(64, 1)
        #print(f'dones shape: {dones.shape}, actions shape: {actions.shape}, rewards shape: {rewards.shape}')
        
        dones = dones.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        #print(f'dones shape after unsqueeze: {dones.shape}, actions shape: {actions.shape}, rewards shape: {rewards.shape}')
        # Krytyk - target Q
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards + self.DISCOUNT * target_q # (~dones) - nie ppotrzebuje bo to time series

        current_q = self.critic(states, actions)
        critic_loss = self.loss_fn(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Soft update target sieci
        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)
        

    def _soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1.0 - self.TAU) * target_param.data)

    def get_action(self, state, noise_std=0.1):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()
        # noise = np.random.normal(0, self.noise_std, size=action.shape)
        # out = self.noise(action)
        # print(out)
        return self.noise(action) #np.clip(action + noise, 0, 1)
    
    def get_action_target(self, state):
        state = state.clone().detach().float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.target_actor(state).cpu().numpy()

        return np.clip(action, 0, 1)


    



EPSILON_DECAY = 0.95

def train_episode(env,trading_desk, episode, epsilon):
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
        #self._get_obs(), reward, done, info
        new_state, reward, done, info = env.step(action)
        #print(env.current_step)
        episode_reward += reward
        
        portfolio_manager.update_replay_memory((current_state, action, reward, new_state, done))
        
        portfolio_manager.train(done)
        current_state = new_state
        step += 1
 
    if not episode % 2:
            print(f"Episode: {episode} Total Reward: {env.total_portfolio} Epsilon: {epsilon:.2f}")
    
    # print(portfolio_manager.noise_std)
    # if portfolio_manager.noise_std > portfolio_manager.MIN_NOISE:
    #     portfolio_manager.noise_std *= portfolio_manager.NOISE_DECAY

    return episode_reward





#tickers = ['AAPL','GOOGL', 'CCL', 'NVDA', 'LTC', 'AMZN']

tickers =  ['NVDA', 'MSFT', 'AAPL', 'GOOG', 'AMZN',
    'META', 'AVGO', 'TSLA', 'JPM',
    'WMT', 'V', 'ORCL', 'LLY', 'NFLX',
    'MA', 'XOM', 'JNJ'  
    ]
#tickers = ["CLFD","IRS","BRC","TBRG","CCNE","CVEO",'AAPL','GOOGL', 'CCL', 'NVDA', 'LTC', 'AMZN']
#tickers = ["CLFD","IRS","BRC","TBRG","CCNE","CVEO"]
trading_desk = {}
data = pd.DataFrame()
min_size = 9999999
for ticker in tickers:
    trader = DQNAgent(ticker)
    trader.load_dqn_agent()
    
    train_data, valid_data, test_data = read_stock_data(ticker)
    training_set = pd.concat([train_data, valid_data, test_data])

    min_size = min(min_size, len(training_set))

    temp = pd.DataFrame(training_set['close'].copy()).rename(columns={'close': ticker})

    temp = temp[:min_size].reset_index(drop=True)
    data = data[:min_size].reset_index(drop=True)    
    data = pd.concat([data[:min_size], temp[:min_size]], axis=1)
    trading_desk[ticker] = trader
    
reward_all = []
evaluate_revards = []
portfolio_manager = AgentPortfolio(input_dim=96, action_dim=len(tickers))
epsilon = 1


data_split = int(len(data)  * 0.8)

train_data = data[:data_split]
valid_data = data[data_split:]

WINDOW_SIZE = 96
env = PortfolioEnv(train_data, window_size=WINDOW_SIZE)
valid_env = PortfolioEnv(valid_data,window_size=WINDOW_SIZE)

print('initilized')
#super dla 200, batch64
EPISODES = 50
MIN_EPSILON = 0.001

max_portfolio_manager = None
max_reward = 0
evaluate_every = 1
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    reward = train_episode(env,trading_desk, episode,epsilon)
    
    
    print("Renderuje środowisko walidacyjne")
    valid_env.reset()
    reward_valid_dataset, steps, info = evaluate_steps_portfolio(valid_env,trading_desk, portfolio_manager)

    render_portfolio_summary(valid_env)
    
    print(info, portfolio_manager.noise.sigma)
    evaluate_revards.append(info)
