import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=8, num_layers=1):
        super(FeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.output_dim = hidden_dim

    def forward(self, x):
        # x: [batch, seq_len=96, input_dim=6]
        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :]            # ostatni krok czasowy
        h = self.dropout(self.relu(h))
        return h                          # [batch, hidden_dim]



class Actor(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 8)
        self.logit_net = nn.Linear(8, action_dim)

    def forward(self, features):
        x = F.relu(self.fc1(features))
        logits = self.logit_net(x)
        return logits



class Critic(nn.Module):
    def __init__(self, feature_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 8)
        self.value = nn.Linear(8, 1)

    def forward(self, features):
        x = F.relu(self.fc1(features))
        #x = F.relu(self.fc2(x))
        v = self.value(x)
        return v.squeeze(-1)


class TraderPPO:
    def __init__(self, input_dim=11, action_dim=3, device='cuda'):
        self.device = device
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_eps = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.epochs = 10
        self.MINIBATCH_SIZE = 128
        self.max_grad_norm = 0.5
        self.replay_memory = []
        self.MIN_REPLAY_MEMORY_SIZE = 500
        
        # sieci
        self.feature_extractor = FeatureExtractor(input_dim).to(device)
        self.actor = Actor(self.feature_extractor.output_dim, action_dim).to(device)
        self.critic = Critic(self.feature_extractor.output_dim).to(device)

        # optymalizatory
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + 
            list(self.actor.parameters()) + 
            list(self.critic.parameters()), 
            lr=3e-4
        )
        # bufor
        self.reset_buffer()

    def reset_buffer(self):
        self.replay_memory = []

    def select_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            features = self.feature_extractor(state_t)
            logits = self.actor(features)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action).detach().cpu().item()
            value = self.critic(features).detach().cpu().item()
        return int(action.item()), float(logp), float(value)

    def store_transition(self, state, action, reward, done, logp, value):
        self.replay_memory.append((state, action, reward, done, logp, value))
        
    def compute_gae(self, rewards, dones, values, last_value=0.0):
        advs = []
        returns = []
        gae = 0.0
        
        # Append last_value for bootstrapping
        values_list = values.tolist() + [last_value]
        
        for t in reversed(range(len(rewards))):
            next_value = values_list[t + 1]
            current_value = values_list[t]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - current_value
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            
            advs.insert(0, gae)
            returns.insert(0, gae + current_value)
        
        advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        
        return advs, returns
    
    def train(self):
        states, actions, rewards, dones, old_logps, values = zip(*self.replay_memory)
        states = torch.tensor(np.array(states, dtype=np.float32), device=self.device)
        actions = torch.tensor(np.array(actions, dtype=np.int64), device=self.device)
        rewards = torch.tensor(np.array(rewards, dtype=np.float32), device=self.device)
        dones = torch.tensor(np.array(dones, dtype=np.float32), device=self.device)
        old_logps = torch.tensor(np.array(old_logps, dtype=np.float32), device=self.device)
        values = torch.tensor(np.array(values, dtype=np.float32), device=self.device)
        
        last_value = 0.0
        if not dones[-1]:  # If the last state was NOT terminal
            with torch.no_grad():
                last_state_t = states[-1].unsqueeze(0)
                features = self.feature_extractor(last_state_t)
                last_value = self.critic(features).item()
        
        advs, returns = self.compute_gae(rewards, dones, values, last_value=last_value)
        for _ in range(self.epochs):
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), self.MINIBATCH_SIZE):
                end = start + self.MINIBATCH_SIZE
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_logps = old_logps[batch_idx]
                batch_advs = advs[batch_idx].detach()
                batch_returns = returns[batch_idx]

                features = self.feature_extractor(batch_states)
                logits = self.actor(features)
                dist = torch.distributions.Categorical(logits=logits)
                new_logps = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logps - batch_old_logps)
                surr1 = ratio * batch_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advs
                policy_loss = -torch.min(surr1, surr2).mean()

                values_pred = self.critic(features)
                value_loss = (batch_returns - values_pred).pow(2).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.feature_extractor.parameters()) +
                    list(self.actor.parameters()) +
                    list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
        self.replay_memory = []
        return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "entropy": entropy.item()}