import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import math


# ===========================
# Noisy Linear Layer
# ===========================
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)

# class RainbowDQN(nn.Module):
#     def __init__(self, input_dim, output_dim, seq_len=256):
#         super(RainbowDQN, self).__init__()
#         self.output_dim = output_dim
#         self.seq_len = seq_len
        
#         # 1D Convolutional layers for temporal feature extraction
#         # Input shape: (batch, seq_len, input_dim) -> transpose to (batch, input_dim, seq_len)
#         self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.2)
        
#         # Calculate flattened size after convolutions
#         # seq_len remains the same due to padding=1
#         self.flatten_size = 16 * seq_len
        
#         # Dueling streams
#         self.adv1 = NoisyLinear(self.flatten_size, 64)
#         self.adv2 = NoisyLinear(64, output_dim)
        
#         self.val1 = NoisyLinear(self.flatten_size, 64)
#         self.val2 = NoisyLinear(64, 1)
    
#     def forward(self, x):
#         # x shape: (batch, seq_len, input_dim)
#         # Transpose for conv1d: (batch, input_dim, seq_len)
#         x = x.transpose(1, 2)
        
#         # Convolutional layers
#         x = self.relu(self.conv1(x))
#         x = self.dropout(x)
        
#         x = self.relu(self.conv2(x))
#         x = self.dropout(x)
        
#         x = self.relu(self.conv3(x))
#         x = self.dropout(x)
        
#         # Flatten
#         x = x.reshape(x.size(0), -1)
        
#         # Dueling streams
#         adv = self.relu(self.adv1(x))
#         adv = self.adv2(adv)
        
#         val = self.relu(self.val1(x))
#         val = self.val2(val)
        
#         # Combine value and advantage
#         q = val + adv - adv.mean(dim=1, keepdim=True)
        
#         return q
    
#     def reset_noise(self):
#         for name, module in self.named_modules():
#             if isinstance(module, NoisyLinear):
#                 module.reset_noise()

class RainbowDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RainbowDQN, self).__init__()
        self.output_dim = output_dim

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=32, batch_first=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        # Dueling streams
        self.adv1 = NoisyLinear(32, 8)
        self.adv2 = NoisyLinear(8, output_dim)

        self.val1 = NoisyLinear(32, 8)
        self.val2 = NoisyLinear(8, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        #print("LSTM output shape:", lstm_out.shape)
        lstm_out = lstm_out[:, -1, :]  # tylko ostatni krok sekwencji
        lstm_out = self.relu(lstm_out)
        lstm_out = self.dropout(lstm_out)

        adv = self.relu(self.adv1(lstm_out))
        adv = self.adv2(adv)

        val = self.relu(self.val1(lstm_out))
        val = self.val2(val)

        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q

    def reset_noise(self):
        for name, module in self.named_modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# ===========================
# Prioritized Replay Buffer
# ===========================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.8):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        return batch, indices, torch.tensor(weights)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


# ===========================
# Rainbow Agent
# ===========================
class RainbowAgent:
    def __init__(self, ticker):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ticker = ticker
        self.filename = f'models/{self.ticker}_rainbow.pt'

        self.observation_space = 11
        self.action_space = 3

        self.model = RainbowDQN(self.observation_space, self.action_space).to(self.device)
        self.target_model = RainbowDQN(self.observation_space, self.action_space).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

        self.memory = PrioritizedReplayBuffer(200_000)
        self.batch_size = 128
        self.gamma = 0.99
        self.beta_start = 0.8
        self.beta_frames = 100_000
        self.frame = 1
        self.update_target_every = 2000
        self.learn_step = 0

    def update_replay_memory(self, transition):
        self.memory.push(transition)

    def compute_td_error(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_q = self.model(next_states)
            next_actions = torch.argmax(next_q, dim=1)
            next_q_target = self.target_model(next_states)
            max_next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_error = target_q - current_q
        return td_error

    def train(self):
        if len(self.memory.buffer) < self.batch_size:
            return

        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        batch, indices, weights = self.memory.sample(self.batch_size, beta)

        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = weights.to(self.device)

        td_errors = self.compute_td_error(states, actions, rewards, next_states, dones)
        loss = (self.loss_fn(td_errors, torch.zeros_like(td_errors)) * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update priorities
        new_priorities = torch.abs(td_errors.detach()).cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, new_priorities)

        # update target network
        self.learn_step += 1
        if self.learn_step % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        self.model.reset_noise()
        self.target_model.reset_noise()
        self.frame += 1

    def get_action(self, state):
        state_v = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_v)
        action = torch.argmax(q_values, dim=1).item()
        return action

    def save(self):
        torch.save({
            'model': self.model.state_dict(),
            'target': self.target_model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, self.filename)

    def load(self):
        checkpoint = torch.load(self.filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.target_model.load_state_dict(checkpoint['target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"✅ Załadowano Rainbow DQN z {self.filename}")
