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
from eval_models import evaluate_steps, render_env_ddpg
from enviroments import TimeSeriesEnvOHLC
import os


# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim):  # state_dim = [n_assets, features] = [4, 97]
#         super(Actor, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv1d(in_channels=96, out_channels=32, kernel_size=1),  # 97 = liczba cech
#             nn.ReLU(),
#             nn.Conv1d(32, 8, kernel_size=1),
#             nn.ReLU()
#         )
#         self.fc = nn.Sequential(
#             nn.Flatten(),                     # [B, 64, 4] → [B, 64*4]
#             nn.Linear(8 * 4, 8),
#             nn.ReLU(),
#             nn.Linear(8, 1),        # action_dim = liczba alokacji
#             nn.Tanh()                # ładne rozkłady alokacji
#         )

#     def forward(self, state):
#         #print(state.shape)
#         x = state.permute(0, 2, 1)  # [B, 4, 97] → [B, 97, 4]
#         x = self.conv(x)            # [B, 64, 4]
#         x = self.fc(x)              # [B, 4]
#         return x
        
# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Critic, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv1d(in_channels=96, out_channels=32, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv1d(32, 8, kernel_size=1),
#             nn.ReLU()
#         )
#         self.fc = nn.Sequential(
#             nn.Flatten(),                            # [B, 64, 4] → [B, 256]
#             nn.Linear(8 * 4 + action_dim, 8),      # dodajemy akcje
#             nn.ReLU(),
#             nn.Linear(8, 1)
#         )

#     def forward(self, state, action):
#         x = state.permute(0, 2, 1)                   # [B, 97, 4]
#         x = self.conv(x)                             # [B, 64, 4]
#         x = x.flatten(start_dim=1)                   # [B, 256]
#         x = torch.cat([x, action], dim=1)            # [B, 256 + 4]
#         x = self.fc(x)                               # [B, 1]
#         return x.flatten()
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):  # state_dim = [n_assets, features] = [4, 97]
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(input_size=96, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
            nn.Tanh()  
        )

    def forward(self, state):
        # state: [B, 4, 97] — traktujemy 4 aktywa jako "czas", 96 cech na każde aktywo
        lstm_out, _ = self.lstm(state)         # lstm_out: [B, 4, 32]
        x = lstm_out[:, -1, :]                 # weź ostatni timestep: [B, 32]
        x = self.fc(x)                         # [B, action_dim]
        return x    
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(input_size=96, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(32 + action_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # wartość Q
        )

    def forward(self, state, action):
        lstm_out, _ = self.lstm(state)        # [B, 4, 8]
        x = lstm_out[:, -1, :]
        #print(x.shape, action.shape)
        x = torch.cat([x, action], dim=1)  # [B, 32 + action_dim]
        x = self.fc(x)
        return x.flatten()
# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim):  # state_dim = [time_steps, features] = [96, 4]
#         super(Actor, self).__init__()
#         # Smaller hidden size for efficiency - now 4 features per timestep
#         self.lstm = nn.LSTM(input_size=4, hidden_size=16, num_layers=1, batch_first=True)
#         # Reordered network layers
#         self.fc = nn.Sequential(
#             nn.Tanh(),                      # Activation first
#             nn.Linear(16, 8),               # Then linear layer
#             nn.ReLU(),                      # Another activation
#             nn.Linear(8, action_dim),       # Final output layer
#             nn.Dropout(0.2),                 # Dropout last
#             nn.Tanh()                      # Final activation for action output
#         )

#     def forward(self, state):
#         # state: [B, 96, 4] — treat 96 as "time", 4 features per timestep  
#         lstm_out, _ = self.lstm(state)         # lstm_out: [B, 96, 16]
#         x = lstm_out[:, -1, :]                 # take last timestep: [B, 16]
#         x = self.fc(x)                         # [B, action_dim]
#         return x    

# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Critic, self).__init__()
#         # Smaller hidden size matching Actor - now 4 features per timestep
#         self.lstm = nn.LSTM(input_size=4, hidden_size=16, num_layers=1, batch_first=True)
#         # Reordered network layers
#         self.fc = nn.Sequential(
#             nn.ReLU(),                         # Activation first
#             nn.Linear(16 + action_dim, 12),    # Then linear layer
#             nn.Dropout(0.2),                   # Dropout in middle
#             nn.Linear(12, 4),                  # Another linear layer
#             nn.ReLU(),                         # Another activation
#             nn.Linear(4, 1)                    # Q-value output last
#         )

#     def forward(self, state, action):
#         lstm_out, _ = self.lstm(state)        # [B, 96, 16]
#         x = lstm_out[:, -1, :]                # [B, 16] - take last timestep
#         x = torch.cat([x, action], dim=1)     # [B, 16 + action_dim]
#         x = self.fc(x)                        # [B, 1]
#         return x.flatten()                    # [B]


class AgentTrader:
    def __init__(self, input_dim=96, action_dim=1): #27 * 4
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

        self.replay_memory = deque(maxlen=50000)
        self.MIN_REPLAY_MEMORY_SIZE = 500
        self.MINIBATCH_SIZE = 64
        self.DISCOUNT = 0.999
        self.TAU = 1e-3  # do soft update

        self.noise_std = 0.3
        self.NOISE_DECAY = 0.9
        self.MIN_NOISE = 0.05
        
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
        
        #actions = np.array([action['portfolio_manager'] for action in actions])
        #print(actions.shape)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        #dones = dones.unsqueeze(1).expand(64, 1)
        # Krytyk - target Q
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            #print(target_q.shape, rewards.shape, dones.shape)
            target_q = rewards + (~dones) * self.DISCOUNT * target_q

        current_q = self.critic(states, actions)
        #print(current_q.shape, target_q.shape)
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
        noise = np.random.normal(0, self.noise_std, size=action.shape)
        return np.clip(action + noise, -1, 1).flatten()
    
    def get_action_target(self, state):
        state = state.clone().detach().float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.target_actor(state).cpu().numpy()

        return np.clip(action, -1, 1).flatten()


    



EPSILON_DECAY = 0.95

def train_episode(env, episode, epsilon):
    episode_reward = 0
    step = 1

    
    current_state = env.reset()
    #print(current_state)
    #print(current_state.shape)
    done = False
    while not done:

        # if np.random.rand() < epsilon:
        #     action_allocation_percentages = torch.rand((1,), dtype=torch.float32)
        #     action_allocation_percentages = action_allocation_percentages.squeeze(0).cpu().numpy()  # [n_assets]
        # else:
        #print(current_state)
        action = trader.get_action(current_state)
        #print(action)
        #print(action.shape)
        #print(current_state)
        new_state, reward, done = env.step(action)
        

        episode_reward += reward
        
        trader.update_replay_memory((current_state, action, reward, new_state, done))
        
        #if np.random.random() >= .7:
        trader.train(done)

        current_state = new_state
        step += 1
 
    if not episode % 2:
            print(f"Episode: {episode} Total Reward: {env.total_profit} Epsilon: {epsilon:.2f}")
    
    print(trader.noise_std)
    if trader.noise_std > trader.MIN_NOISE:
        trader.noise_std *= trader.NOISE_DECAY

    return episode_reward




#ticker = 'AAPL'
#train_df, val_df ,rl_df,test_df = read_stock_data(ticker)
#training_set = pd.concat([train_df, val_df ,rl_df,test_df])
#print(training_set[['open', 'high', 'low', 'close']])


#print(data)
reward_all = []
evaluate_revards = []
trader = AgentTrader()
epsilon = 1

data = np.sin(np.linspace(0, 500, 2500)).astype(np.float32) + 1


window_size = 10  # number of data points per OHLC bar

data = np.sin(np.linspace(0, 500, 50000)).astype(np.float32) + 1

# Parametr: ziarnistość OHLC (co ile punktów tworzymy nowy słupek)
grain = 10

# Upewniamy się, że liczba danych jest wielokrotnością grain
data = data[:len(data) // grain * grain]

# Grupujemy dane po 'grain' elementów
data_reshaped = data.reshape(-1, grain)

# Tworzymy OHLC DataFrame
ohlc = pd.DataFrame({
    'open': data_reshaped[:, 0],
    'high': data_reshaped.max(axis=1),
    'low': data_reshaped.min(axis=1),
    'close': data_reshaped[:, -1],
})

data = ohlc
# data = training_set[['open', 'high', 'low', 'close']] #.values.astype(np.float32)

data_split = int(len(data)  * 0.8)

train_data = data[:data_split]
valid_data = data[data_split:]

env = TimeSeriesEnvOHLC(train_data,96)
valid_env = TimeSeriesEnvOHLC(valid_data, 96)


WINDOW_SIZE = 96

EPISODES = 15
MIN_EPSILON = 0.001

max_portfolio_manager = None
max_reward = 0
evaluate_every = 1
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    reward = train_episode(env, episode,epsilon)
    
    #reward_all.append(reward)
    #if episode % evaluate_every:
    #render_env(valid_env)
    #if not episode % 1:
    
    # if max_portfolio_manager:
    #     reward_valid_dataset, steps, info = evaluate_steps_portfolio(valid_env, trader, max_portfolio_manager)
    # else:
    
    print("Renderuje środowisko walidacyjne")
    valid_env.reset()
    total_reward = evaluate_steps(valid_env, trader,OHCL=True)

    #render_env(valid_env)
    render_env_ddpg(valid_env, OHCL=True)
    
    #print(env.portfolio_value_history)
    print(total_reward)
    #print(info)
    evaluate_revards.append(total_reward)
    
    # if reward_valid_dataset > max_reward and episode > 1:
    #     max_reward = reward_valid_dataset
    #     #print(max_reward)
    #     max_portfolio_manager = deepcopy(portfolio_manager)
    
    # #nadpisz jeśli się pogorszy
    # if max_reward > 0 and episode > 1 and reward_valid_dataset / max_reward <= .7:
    #     agent = deepcopy(max_portfolio_manager)
