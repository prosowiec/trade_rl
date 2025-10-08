import torch.nn as nn
import torch.optim as optim
import torch
from collections import deque
import random
import numpy as np
import logging


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.out_steps = output_dim

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=8, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(8, output_dim)

    def forward(self, x):
        #_, (h_n, _) = self.lstm(x)  # h_n: [1, batch, lstm_units]
        h_n, _ = self.lstm(x) 
        h_n = h_n.squeeze(0)
        h_n = self.relu(h_n)
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

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-2)
        self.loss_fn = nn.MSELoss()


        self.REPLAY_MEMORY_SIZE = 5000000
        self.MIN_REPLAY_MEMORY_SIZE = 500 
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0
        
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
        
        return action

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
        logging.info(f"Model załadowany z {self.filename}")
