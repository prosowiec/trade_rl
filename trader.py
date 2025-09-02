import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from copy import deepcopy
from torch.distributions import Categorical
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from dataOps import get_training_set_from_IB
from source.database import upload_stock_data, read_stock_data, upsert_training_logs
from enviroments import TimeSeriesEnv_simple
from eval_models import evaluate_steps, render_env

import os
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"

class DQN(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(DQN, self).__init__()
		self.out_steps = output_dim

		self.lstm = nn.LSTM(input_size=input_dim, hidden_size=8, batch_first=True)
		self.dropout = nn.Dropout(p=0.2)
		self.fc4 = nn.Linear(8, output_dim)

	def forward(self, x):
		#_, (h_n, _) = self.lstm(x)  # h_n: [1, batch, lstm_units]
		h_n, _ = self.lstm(x) 
		h_n = h_n.squeeze(0)

		x = self.fc4(h_n)
		x = x.view(-1, self.out_steps, 1)
		return x


class DQNAgent:
    def __init__(self, ticker):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ticker = ticker
        self.filename = f'models/{self.ticker}_trader'

        self.observarion_space = 96
        self.action_space = 3
        
        self.model = DQN(self.observarion_space, self.action_space).to(self.device)
        self.target_model = DQN(self.observarion_space, self.action_space).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        #self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-2)
        self.loss_fn = nn.MSELoss()


        self.REPLAY_MEMORY_SIZE = 5000000
        self.MIN_REPLAY_MEMORY_SIZE = 500 
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0
        
        #model setings
        self.UPDATE_TARGET_EVERY = 2
        self.MINIBATCH_SIZE = 128
        self.DISCOUNT = 0.99
        
        self.AGGREGATE_STATS_EVERY = 10
        
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state):
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

        # Rozpakowanie danych
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states_v = torch.from_numpy(np.array(states)).float().to(self.device)
        next_states_v = torch.from_numpy(np.array(next_states)).float().to(self.device)
        actions_v = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards_v = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_v = torch.tensor(dones, dtype=torch.bool, device=self.device)        

        with torch.no_grad():
            target_qs = self.target_model(next_states_v).flatten(start_dim=1)
            max_future_qs = torch.max(target_qs, dim=1)[0]
            new_qs = rewards_v + (~dones_v * self.DISCOUNT * max_future_qs)

        current_qs = self.model(states_v).flatten(start_dim=1)
        predicted_qs = current_qs.gather(1, actions_v.unsqueeze(1)).squeeze()

        loss = self.loss_fn(predicted_qs, new_qs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0

    def get_qs(self, state, target_model=False):
        state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if target_model == False:
                qs = self.model(state_v)
            else:
                qs = self.target_model(state_v)
        return qs 
    
    def get_action(self, state, target_model = False):
        qs = self.get_qs(state, target_model)
        action = torch.argmax(qs).item()
        
        return action #.cpu().numpy()[0]

    def reset_agent_weights(self):
        self.model = DQN(self.observarion_space, self.action_space).to(self.device)
        self.target_model = DQN(self.observarion_space, self.action_space).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def save_dqn_agent(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.filename)

        
    def load_dqn_agent(self):
        checkpoint = torch.load(self.filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model załadowany z {self.filename}")


def train_episode(agent : DQNAgent,env: TimeSeriesEnv_simple, epsilon):
    episode_reward = 0
    step = 1

    current_state = env.reset()
    done = False
    while not done:
        #print(current_state)
        if np.random.random() > epsilon:
            probs = torch.softmax(agent.get_qs(current_state).squeeze(), dim=0)
            dist = Categorical(probs)
            action = dist.sample().item()
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done = env.step(action)

        episode_reward += reward
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        
        if np.random.random() >= .7:
            agent.train(done)

        current_state = new_state
        step += 1
 
    return episode_reward


def prepare_environments(ticker, newData = False, window_size=96):
    if newData:
        training_set = get_training_set_from_IB(ticker)
        upload_stock_data(training_set)
        
    train_data, valid_data, test_data = read_stock_data(ticker)
    env = TimeSeriesEnv_simple(train_data['close'].values, window_size=window_size)
    valid_env = TimeSeriesEnv_simple(valid_data['close'].values, window_size=window_size)
    test_env = TimeSeriesEnv_simple(test_data['close'].values, window_size=window_size)

    return env, valid_env, test_env




def train_trader(ticker, newData = False):
    env, valid_env, test_env = prepare_environments(ticker, newData)

    agent = DQNAgent(ticker)
    max_agent = DQNAgent(ticker)
    max_reward = 0

    reward_all = []
    evaluate_rewards = []
    test_rewards = []

    epsilon = 1  # not a constant, going to be decayed
    MIN_EPSILON = 0.01
    EPISODES = 100

    EPSILON_DECAY = (epsilon - MIN_EPSILON) / EPISODES #0.975

    render_test_every = 10
    weight_reset = False
    for episode in range(1, EPISODES + 1):
        reward = train_episode(agent,env, epsilon)
                
        if epsilon > MIN_EPSILON:
            epsilon -= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
        
        reward_all.append(reward)
        valid_env.reset()
        test_env.reset()
        reward_valid_env = evaluate_steps(valid_env, agent.target_model)
        reward_test_env = evaluate_steps(test_env, agent.target_model)
        evaluate_rewards.append(reward_valid_env)
        test_rewards.append(reward_test_env)
        
        if reward_valid_env > max_reward:
            max_reward = reward_valid_env
            max_agent = deepcopy(agent)
            max_agent.save_dqn_agent()
        
        if max_reward > 0 and episode > 10 and reward_valid_env / max_reward <= .5:
            agent = deepcopy(max_agent)
        
        if not episode % render_test_every:
            render_env(test_env)

        print(f"Episode: {episode:<5} | "
            f"Reward: {reward:<10.4f} | "
            f"Valid Reward: {reward_valid_env:<10.4f} | "
            f"Test Reward: {reward_test_env:<10.4f} | "
            f"Max Reward: {max_reward:<10.4f} | "
            f"Epsilon: {epsilon:<6.2f}")
        
        if epsilon < 0.5 and np.mean(evaluate_rewards[-5:]) == reward_valid_env:
            break
        
        if not weight_reset and len(reward_all) >= 5 and np.mean(evaluate_rewards[-5:]) == 0:
            print('Reseting weights')
            agent.reset_agent_weights()
            weight_reset = True
        
        if (len(reward_all) > 10 and np.mean(reward_all[-10:]) == 0) or (max_reward == 0 and episode > 10):
            return [],[],[]
        
    return reward_all, evaluate_rewards, test_rewards

def trining_retry_loop(ticker, newData=False, num_retries=15):
    reward_all, evaluate_rewards, test_rewards = [],[],[]
    retry = 0
    while not reward_all and retry < num_retries:
        reward_all, evaluate_rewards, test_rewards = train_trader(ticker, newData=newData)
        if not reward_all:
            retry +=1
            seed = random.randint(0, 1_000_000)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            print(f'--------------- Retrying {retry} ---------------')
    
    return reward_all, evaluate_rewards, test_rewards

if __name__=="__main__":
    #tickers = ['AAPL','GOOGL', 'CCL', 'NVDA', 'LTC', 'AMZN']
    tickers = ["CLFD","IRS","BRC","TBRG","CCNE","CVEO"]
    
    for ticker in tickers:
        reward_all, evaluate_rewards, test_rewards = trining_retry_loop(ticker, newData = True)
        training_log_df = upsert_training_logs(reward_all, evaluate_rewards, test_rewards,ticker)
