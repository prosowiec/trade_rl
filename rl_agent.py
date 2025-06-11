import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import time

from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2


DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 1000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 200  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 128  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 10  # Terminal states (end of episodes)
MODEL_NAME = 'window48_profit'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 1000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.9575
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 100  # episodes
SHOW_PREVIEW = True

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)



class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.output = nn.Linear(8, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.output(x)

class DQNAgent:
    def __init__(self, observation_space,action_space):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(observation_space, action_space).to(self.device)
        self.target_model = DQN(observation_space,action_space).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Rozpakowanie danych
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states_v = torch.from_numpy(np.array(states)).float().to(self.device)
        next_states_v = torch.from_numpy(np.array(next_states)).float().to(self.device)
        actions_v = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards_v = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_v = torch.tensor(dones, dtype=torch.bool, device=self.device)        
        
        with torch.no_grad():
            target_qs = self.target_model(next_states_v)
            max_future_qs = torch.max(target_qs, dim=1)[0]
            new_qs = rewards_v + (~dones_v * DISCOUNT * max_future_qs)

        current_qs = self.model(states_v)
        predicted_qs = current_qs.gather(1, actions_v.unsqueeze(1)).squeeze()

        loss = self.loss_fn(predicted_qs, new_qs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0

    def get_qs(self, state):
        #state = np.array(state) / 255.0
        state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qs = self.model(state_v)
        return qs.cpu().numpy()[0]


def train_episode(agent,env, episode, epsilon):

    episode_reward = 0
    step = 1

    current_state = env.reset()

    done = False
    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward


        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step, env.min_val, env.max_val)

        current_state = new_state
        step += 1
 
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            print(f"Episode: {episode} Total Reward: {env.total_profit} Epsilon: {epsilon:.2f}")

    
    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    return episode_reward


def save_dqn_agent(agent, filename="dqn_model.pth"):
    torch.save({
        'model_state_dict': agent.model.state_dict(),
        'target_model_state_dict': agent.target_model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
    }, filename)



def train_dqn_agent(agent,env):
    
    EPISODES = 400
    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        train_episode(agent,env,episode,epsilon)

    save_dqn_agent(agent, "trained_agent_stock.pth")


def load_dqn_agent(agent, filename="dqn_model.pth"):
    checkpoint = torch.load(filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.target_model.load_state_dict(checkpoint['target_model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Model załadowany z {filename}")

def train_parallel_episode(agent, envs, episode, epsilon, NUM_ENVS = 4):
    states, _ = envs.reset()
    episode_rewards = [0.0 for _ in range(NUM_ENVS)]
    dones = [False for _ in range(NUM_ENVS)]
    step = 1

    while not all(dones):
        actions = []
        for i in range(NUM_ENVS):
            if np.random.random() > epsilon:
                actions.append(np.argmax(agent.get_qs(states[i])))
            else:
                actions.append(np.random.randint(0, envs.single_action_space.n))
        
        next_states, rewards, terminated, truncated, infos = envs.step(actions)

        for i in range(NUM_ENVS):
            done = terminated[i] or truncated[i]
            agent.update_replay_memory((states[i], actions[i], rewards[i], next_states[i], done))
            episode_rewards[i] += rewards[i]
            if not dones[i]:
                dones[i] = done

        agent.train(any(dones), step)
        states = next_states
        step += 1

    avg_reward = sum(episode_rewards) / NUM_ENVS
    ep_rewards.append(avg_reward)

    if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
        print(f"Episode: {episode} Avg Reward: {avg_reward:.2f} Epsilon: {epsilon:.3f}")

    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    return avg_reward

def train_dqn_agent_parallel(agent, envs):
    global epsilon
    EPISODES = 400

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        train_parallel_episode(agent, envs, episode, epsilon)

    save_dqn_agent(agent, "trained_agent_parallel.pth")
