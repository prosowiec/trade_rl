# A3C z obsługą GPU
# Mocno inspirowane Morvan Zhou (github.com/MorvanZhou/pytorch-A3C)

import gym
import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
            weight_decay=0, device = 'cuda'):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data).to(device)
                state['exp_avg_sq'] = T.zeros_like(p.data).to(device)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99, device = 'cuda'):
        super(ActorCritic, self).__init__()

        self.gamma = gamma

        self.pi1 = nn.Linear(*input_dims, 128)
        self.v1 = nn.Linear(*input_dims, 128)
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)

        self.rewards = []
        self.actions = []
        self.states = []
        self.device = device
        self.to(device)

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def forward(self, state):
        pi1 = F.relu(self.pi1(state))
        v1 = F.relu(self.v1(state))

        pi = self.pi(pi1)
        v = self.v(v1)

        return pi, v

    def calc_R(self, done):
        states = T.tensor(np.array(self.states), dtype=T.float, device=self.device)
        _, v = self.forward(states)

        R = v[-1] * (1 - int(done))
        batch_return = []

        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.stack(batch_return).to(self.device)

        return batch_return

    def calc_loss(self, done):
        states = T.tensor(np.array(self.states), dtype=T.float, device=self.device)
        actions = T.tensor(self.actions, dtype=T.long, device=self.device)

        returns = self.calc_R(done)
        pi, values = self.forward(states)
        values = values.squeeze()

        critic_loss = (returns - values) ** 2

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs * (returns - values)

        total_loss = (critic_loss + actor_loss).mean()
        return total_loss

    def choose_action(self, observation):
        state = T.from_numpy(observation).float().unsqueeze(0).to(self.device)
        pi, _ = self.forward(state)
        probs = T.softmax(pi, dim=1)
        action = probs.argmax(dim=1).item()
        return action

class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions, 
                gamma, lr, name, global_ep_idx, env_id, N_GAMES=3000, device = 'cuda'):
        super(Agent, self).__init__()
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%02i' % name
        self.episode_idx = global_ep_idx
        self.env = gym.make(env_id)
        self.optimizer = optimizer
        self.N_GAMES = N_GAMES
        self.T_MAX = 5
        self.device = device

    def run(self):
        t_step = 1
        while self.episode_idx.value < self.N_GAMES:
            done = False
            observation, _ = self.env.reset()
            score = 0
            self.local_actor_critic.clear_memory()

            while not done:
                action = self.local_actor_critic.choose_action(observation)
                observation_, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.local_actor_critic.remember(observation, action, reward)
                observation = observation_
                score += reward
                t_step += 1

                if done or t_step % self.T_MAX == 0:
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(
                            self.local_actor_critic.parameters(),
                            self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                    self.optimizer.step()
                    self.local_actor_critic.load_state_dict(
                            self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()
            
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            print(self.name, 'episode', self.episode_idx.value, 'reward %.1f' % score)

if __name__ == '__main__':
    mp.set_start_method('spawn')  # Upewnij się, że multiprocessing działa poprawnie na wszystkich OS
    lr = 1e-4
    env_id = 'CartPole-v0'
    n_actions = 2
    input_dims = [4]
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.share_memory()
    global_actor_critic.to(device)

    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)

    workers = [Agent(global_actor_critic,
                     optim,
                     input_dims,
                     n_actions,
                     gamma=0.99,
                     lr=lr,
                     N_GAMES=3000,
                     name=i,
                     global_ep_idx=global_ep,
                     env_id=env_id) for i in range(mp.cpu_count())]

    [w.start() for w in workers]
    [w.join() for w in workers]
