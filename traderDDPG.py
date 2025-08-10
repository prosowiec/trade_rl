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

from rl_agent_simple import DQNAgent
from rl_agent import load_dqn_agent
from source.database import read_stock_data
from copy import deepcopy
#from enviroments import TimeSeriesEnv_simple
from gym import spaces
from eval_models import evaluate_steps, render_env_ddpg
from enviroments import TimeSeriesEnvOHLC
import os
from torch.distributions import Categorical



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):  # state_dim = [n_assets, features] = [4, 97]
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(input_size=11, hidden_size=1, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(256 + 1, 4),
            nn.ReLU(),
            nn.Linear(4, action_dim),
            nn.Softmax()  
        )
    
    def forward(self, state, position_ratio):
        # state: [B, 4, 97] — now we need to reshape to use 4 as input features
        # Option 1: Transpose to treat features as sequence steps
        state_transposed = state.transpose(1, 2)  # [B, 97, 4]
        lstm_out, _ = self.lstm(state_transposed)  # lstm_out: [B, 97, 32]
        #x = lstm_out[:, -1, :]                     # take last timestep: [B, 32]
        BATCH_SIZE = lstm_out.shape[0]
        x = lstm_out.contiguous().view(BATCH_SIZE,-1)
        #print(f"Out Shape : {lstm_out.shape}, X shape: {x.shape}")
        #print(x.shape, position_ratio.shape)
        x = torch.cat([x, position_ratio], dim=1)  # [B, 32 + 1]
        x = self.fc(x)                             # [B, action_dim]
        return x
   
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(input_size=11, hidden_size=1, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(256 +3+1 , 4),  # +1 for position_ratio
            nn.ReLU(),
            nn.Linear(4, 1)  # Output: Q-value
        )

    def forward(self, state, position_ratio, action):
        # state: [B, 4, 97] — treat features as sequence
        #print(state.shape, action.shape, position_ratio.shape)
        state_transposed = state.transpose(1, 2)  # [B, 97, 4]
        lstm_out, _ = self.lstm(state_transposed)  # [B, 97, 16]
        #x = lstm_out[:, -1, :]                     # [B, 16]
        
        BATCH_SIZE = lstm_out.shape[0]
        x = lstm_out.contiguous().view(BATCH_SIZE,-1)
        # Add action and position_ratio
        print(x.shape, action.shape, position_ratio.shape )
        #print(action)
        x = torch.cat([x, action, position_ratio], dim=1)  # [B, 16 + action_dim + 1]
        x = self.fc(x)  # [B, 1]
        return x.flatten()
    
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.9):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_decay = 0.999995
        self.min_sigma = 0.1
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
        return self.state
    
    def __call__(self, action):
        """Call to sample noise."""
        return np.clip(action.cpu() + self.sample(),0,2)
    
    
class AgentTrader:
    def __init__(self, input_dim=96, action_dim=3): #27 * 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(input_dim, action_dim).to(self.device)
        self.target_actor = Actor(input_dim, action_dim).to(self.device)
        self.critic = Critic(input_dim, action_dim).to(self.device)
        self.target_critic = Critic(input_dim, action_dim).to(self.device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.loss_fn = nn.MSELoss()

        self.replay_memory = deque(maxlen=5000000)
        self.MIN_REPLAY_MEMORY_SIZE = 500
        self.MINIBATCH_SIZE = 64
        self.DISCOUNT = 0.9999
        self.TAU = 1e-3  # do soft update

        self.noisy_action = OUNoise(size=action_dim)
        # self.noise_std = 1
        # self.NOISE_DECAY = 0.5
        # self.MIN_NOISE = 0.05
        
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state):
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        #print(actions)
        states_only = [s[0] for s in states]
        position_ratios = [s[1] for s in states]
        
        states_only_next = [s[0] for s in next_states]
        position_ratiosy_next = [s[1] for s in next_states]


        states = torch.from_numpy(np.array(states_only, dtype=np.float32)).to(self.device)
        position_ratios = torch.from_numpy(np.array(position_ratios, dtype=np.float32)).to(self.device)
        
        next_states = torch.from_numpy(np.array(states_only_next, dtype=np.float32)).to(self.device)
        position_ratios_next = torch.from_numpy(np.array(position_ratiosy_next, dtype=np.float32)).to(self.device)

        actions = torch.from_numpy(np.array(actions, dtype=np.float32)).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        #dones = dones.unsqueeze(1).expand(64, 1)
        # Krytyk - target Q
        with torch.no_grad():
            next_actions = self.target_actor(next_states, position_ratios_next)
            target_q = self.target_critic(next_states,position_ratios_next, next_actions)
            #print(target_q.shape, rewards.shape, dones.shape)
            target_q = rewards + (~dones) * self.DISCOUNT * target_q
        print(actions,actions.shape )
        current_q = self.critic(states,position_ratios,actions)
        #print(current_q.shape, target_q.shape)
        critic_loss = self.loss_fn(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        predicted_actions = self.actor(states,position_ratios)
        actor_loss = -self.critic(states,position_ratios, predicted_actions).mean()
        
        

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

    def get_action(self, state):
        state, position_ratio = state
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        position_ratio = torch.tensor([position_ratio], dtype=torch.float32).to(self.device)
        #print(state.shape, position_ratio.shape)
        with torch.no_grad():
            #action = self.actor(state, position_ratio).cpu().numpy()
            probs = torch.softmax(self.noisy_action(self.actor(state, position_ratio)).squeeze(), dim=0)
            dist = Categorical(probs)
            action = dist.sample().item()  # zamiast .numpy()[0]
        #print(f"Action: {action}")
        return [action]
    
    def get_action_target(self, state):
        state, position_ratio = state
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        position_ratio = torch.tensor([position_ratio], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            #action = self.target_actor(state, position_ratio).cpu().numpy()
            action = torch.argmax(self.target_actor(state, position_ratio)).item()
        print(f"Action: {action}")
        return [action]


    




def train_episode(env, episode, epsilon):
    episode_reward = 0
    step = 1

    
    current_state = env.reset()
    trader.noisy_action.reset()
    done = False
    while not done:

        action = trader.get_action(current_state)
        new_state, reward, done = env.step(action)
        

        episode_reward += reward
        
        trader.update_replay_memory((current_state, action, reward, new_state, done))
        
        trader.train(done)

        current_state = new_state
        step += 1
 
    if not episode % 2:
            print(f"Episode: {episode} Total Reward: {env.total_profit} Epsilon: {epsilon:.2f}")
    

    return episode_reward




ticker = 'CCL'
train_df, val_df ,rl_df,test_df = read_stock_data(ticker)
training_set = pd.concat([train_df, val_df ,rl_df,test_df])
print(training_set[['open', 'high', 'low', 'close']])



data = np.sin(np.linspace(0, 500, 2500)).astype(np.float32) + 1


window_size = 10  # number of data points per OHLC bar

data = np.sin(np.linspace(0, 500, 50000)).astype(np.float32) + 1

#Parametr: ziarnistość OHLC (co ile punktów tworzymy nowy słupek)
grain = 10

#Upewniamy się, że liczba danych jest wielokrotnością grain
data = data[:len(data) // grain * grain]

#Grupujemy dane po 'grain' elementów
data_reshaped = data.reshape(-1, grain)

#Tworzymy OHLC DataFrame
ohlc = pd.DataFrame({
    'open': data_reshaped[:, 0],
    'high': data_reshaped.max(axis=1),
    'low': data_reshaped.min(axis=1),
    'close': data_reshaped[:, -1],
})

data = ohlc
data = training_set[['open', 'high', 'low', 'close', 'volume']] #.values.astype(np.float32)

data_split = int(len(data)  * 0.8)

train_data = data[:data_split]
valid_data = data[data_split:]


WINDOW_SIZE = 256
env = TimeSeriesEnvOHLC(train_data,WINDOW_SIZE)
valid_env = TimeSeriesEnvOHLC(valid_data, WINDOW_SIZE)

#print(data)
reward_all = []
evaluate_revards = []
trader = AgentTrader(WINDOW_SIZE)
epsilon = 1


EPISODES = 15

max_portfolio_manager = None
max_reward = 0
evaluate_every = 1
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    reward = train_episode(env, episode,epsilon)
    
    print("Renderuje środowisko walidacyjne")
    valid_env.reset()
    total_reward = evaluate_steps(valid_env, trader,OHCL=True)

    render_env_ddpg(valid_env, OHCL=True, window_size=WINDOW_SIZE)
    
    print(total_reward)
    evaluate_revards.append(total_reward)
    
