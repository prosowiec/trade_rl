import torch.nn as nn
import torch.optim as optim
import torch
from collections import deque
import random
import numpy as np
import logging
from agents.noise import OUNoise

class Actor(nn.Module):
    def __init__(self,input_dim, action_dim):
        super(Actor, self).__init__()
        self.input_dim = input_dim
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.input_dim, out_channels=32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 2, kernel_size=1),
            nn.ReLU()
            
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                     # [B, 64, 4] → [B, 64*4]
            nn.Linear(2 * action_dim + 2 * action_dim, action_dim),
            #nn.ReLU()
            nn.Softmax(dim=1)
            #nn.Sigmoid()
        )

    def forward(self, state):
        x = state.permute(0, 2, 1)  # [batch, assets, sequence] → [B, sequence, assets]
        trader_actions = x[:, self.input_dim, :].unsqueeze(1)
        portfolio_features = x[:, self.input_dim + 1, :].unsqueeze(1)
        x = x[:, :self.input_dim, :]          # [B, 96, A]

        x = self.conv(x)            # [B, 64, 4]
        x = torch.cat([x, trader_actions, portfolio_features], dim=1)
        x = self.fc(x)              # [B, 4]
        return x
        
class Critic(nn.Module):
    def __init__(self,input_dim, action_dim):
        super(Critic, self).__init__()
        self.input_dim = input_dim
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=96, out_channels=32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 2, kernel_size=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                           
            nn.Linear(2 * action_dim + 3 * action_dim , 1)     
        )

    def forward(self, state, action):
        x = state.permute(0, 2, 1)               
        trader_actions = x[:, self.input_dim, :].unsqueeze(1)   
        portfolio_features = x[:, self.input_dim + 1, :].unsqueeze(1)      
        x = x[:, :self.input_dim, :]          
        action = action.unsqueeze(1)  
        x = self.conv(x)                            
        x = torch.cat([x, action, trader_actions, portfolio_features], dim=1)            
        x = self.fc(x) 
        return x



class AgentPortfolio:
    def __init__(self, ticker_list : list, input_dim=96, action_dim=6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(input_dim, action_dim).to(self.device)
        self.target_actor = Actor(input_dim, action_dim).to(self.device)
        self.critic = Critic(input_dim, action_dim).to(self.device)
        self.target_critic = Critic(input_dim, action_dim).to(self.device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4,  weight_decay=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4, weight_decay=1e-2)

        self.loss_fn = nn.MSELoss()

        self.replay_memory = deque(maxlen=5000000)
        self.MIN_REPLAY_MEMORY_SIZE = 1000
        self.MINIBATCH_SIZE = 128
        self.DISCOUNT = 0.9995
        self.TAU = 1e-4

        self.noise = OUNoise(size=action_dim, mu=0.0, theta=0.15, sigma=0.4)
        ticker_list = "_".join(ticker_list)
        self.filename = 'trade_rl/models/portfolio_manager_' + ticker_list

        
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)

        actions = np.array([action['portfolio_manager'] for action in actions])
        actions = torch.from_numpy(np.array(actions, dtype=np.float32)).to(self.device)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).to(self.device)

        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        
        dones = dones.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        #print(f'dones shape after unsqueeze: {dones.shape}, actions shape: {actions.shape}, rewards shape: {rewards.shape}')
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards + self.DISCOUNT * target_q * (~dones)

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

        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)
        

    def _soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1.0 - self.TAU) * target_param.data)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()
        return self.noise(action)
    
    def get_action_target(self, state):
        #state = state.clone().detach().float().unsqueeze(0).to(self.device)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.target_actor(state).cpu().numpy()

        return np.clip(action.flatten(), 0, 1)
    
    def save_agent(self):
        torch.save({
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
        }, self.filename)
        
    def load_agent(self):
        checkpoint = torch.load(
            self.filename,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

        logging.info(f"Agent załadowany z {self.filename}")