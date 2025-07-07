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
from manager_env import PortfolioEnv
import os

# os.environ['CUDA_LAUNCH_BLOCKING']="1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

# np.random.seed(2137)
# torch.manual_seed(2137)  # dla CPU
# torch.cuda.manual_seed(2137)  # dla GPU (jeśli używasz)
# torch.cuda.manual_seed_all(2137)  # dla wszystkich GPU

# Opcjonalnie: dla pełnej deterministyczności




# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Actor, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim, 6), # Using the larger network from previous advice
#             # nn.ReLU(),
#             # nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(6, action_dim),
#             nn.Sigmoid()  # Zakres [-1, 1] dla każdej akcji
#         )

#     def forward(self, state):
#         return self.net(state)

# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Critic, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim + action_dim, 6),
#             # nn.ReLU(),
#             # nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(6, 1),
#         )

#     def forward(self, state, action):
#         x = torch.cat([state, action], dim=1)
#         return self.net(x)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 48), # Using the larger network from previous advice
            nn.ReLU(),
            nn.Linear(48, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, action_dim),
            nn.Sigmoid()  # Zakres [-1, 1] dla każdej akcji
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
            nn.Dropout(0.3),
            nn.Linear(16, 1),
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

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.loss_fn = nn.MSELoss()

        self.replay_memory = deque(maxlen=50000)
        self.MIN_REPLAY_MEMORY_SIZE = 500
        self.MINIBATCH_SIZE = 64
        self.DISCOUNT = 0.999
        self.TAU = 1e-3  # do soft update

        self.noise_std = 0.2
        self.NOISE_DECAY = 0.8
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
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Aktor - maksymalizacja oceny krytyka
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
            action = self.actor(state).cpu().numpy()[0]
        noise = np.random.normal(0, self.noise_std, size=action.shape)
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
        #print(action_allocation_percentages)
        #action_allocation_percentages = torch.sigmoid(action_allocation_percentages)  # Ensure 0-1 range
        #action_allocation_percentages = action_allocation_percentages.squeeze(0).cpu().numpy()  # [n_assets]
        #print(action_allocation_percentages)
        action = {
            'trader': trader.get_action(env.get_price_window(), target_model = True),
            'portfolio_manager': np.array([action_allocation_percentages]).flatten()
        }
        
        new_state, reward, done, info = env.step(action)
        

        episode_reward += reward
        
        #if not np.isnan(action_allocation_percentages):
        #print(action_allocation_percentages, reward)
        portfolio_manager.update_replay_memory((current_state, action, reward, new_state, done))
        
        #if np.random.random() >= .7:
        portfolio_manager.train(done)
        #print(action_allocation_percentages)
        # else:
        #     print(current_state, reward)
        #     return

        current_state = new_state
        #print(action_allocation_percentages, env.cash)
        step += 1
 
    if not episode % 2:
            print(f"Episode: {episode} Total Reward: {env.total_porfolio} Epsilon: {epsilon:.2f}")
    
    print(portfolio_manager.noise_std)
    if portfolio_manager.noise_std > portfolio_manager.MIN_NOISE:
        portfolio_manager.noise_std *= portfolio_manager.NOISE_DECAY

    return episode_reward



ticker = 'AAPL'
train_df, val_df ,rl_df,test_df = read_stock_data(ticker)
training_set = pd.concat([train_df, val_df ,rl_df,test_df])
#training_set

trader = DQNAgent()
load_dqn_agent(trader, 'aapl_best_agent_vc_dimOPT.pth')
#load_dqn_agent(trader, 'sinus_trader.pth')


reward_all = []
evaluate_revards = []
portfolio_manager = AgentPortfolio()
epsilon = 1


data = training_set['close'].copy()
data[ticker] = training_set['close']
data = pd.DataFrame(data[ticker])

# data = np.sin(np.linspace(0, 500, 2500)).astype(np.float32) + 1
# data = pd.DataFrame(data, columns=["close"])

data_split = int(len(data)  * 0.8)

train_data = data[:data_split]
valid_data = data[data_split:]

WINDOW_SIZE = 96
env = PortfolioEnv(train_data, window_size=WINDOW_SIZE)
valid_env = PortfolioEnv(valid_data,window_size=WINDOW_SIZE)


#super dla 200, batch64
EPISODES = 50
MIN_EPSILON = 0.001

#state = np.random.get_state()
#print(state)

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
    #if not episode % 1:
    
    # if max_portfolio_manager:
    #     reward_valid_dataset, steps, info = evaluate_steps_portfolio(valid_env, trader, max_portfolio_manager)
    # else:
    valid_env.reset()
    reward_valid_dataset, steps, info = evaluate_steps_portfolio(valid_env, trader, portfolio_manager)

    if not episode % 1:
        render_env(valid_env)
    
    #print(env.portfolio_value_history)
    print()
    print(info)
    evaluate_revards.append(info)
    
    # if reward_valid_dataset > max_reward and episode > 1:
    #     max_reward = reward_valid_dataset
    #     #print(max_reward)
    #     max_portfolio_manager = deepcopy(portfolio_manager)
    
    # #nadpisz jeśli się pogorszy
    # if max_reward > 0 and episode > 1 and reward_valid_dataset / max_reward <= .7:
    #     agent = deepcopy(max_portfolio_manager)
