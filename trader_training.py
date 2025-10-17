import numpy as np
import torch
import numpy as np
import random
from copy import deepcopy
from torch.distributions import Categorical
import logging
import os
from agents.traderModel import DQNAgent


from utils.dataOps import get_training_set_from_IB
from utils.database import upload_stock_data, read_stock_data, upsert_training_logs
from agent_env.enviroments import TimeSeriesEnv_simple
from eval.eval_models import evaluate_steps, render_env
from tickers import Tickers

os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)



def train_episode(agent : DQNAgent,env: TimeSeriesEnv_simple, epsilon):
    episode_reward = 0
    step = 1

    current_state = env.reset()
    done = False
    while not done:

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
        # TODO: pass IB api instance
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

    EPSILON_DECAY = (epsilon - MIN_EPSILON) / EPISODES

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

        logging.info(
            f"Episode: {episode:<5} | "
            f"Reward: {reward:<10.4f} | "
            f"Valid Reward: {reward_valid_env:<10.4f} | "
            f"Test Reward: {reward_test_env:<10.4f} | "
            f"Max Reward: {max_reward:<10.4f} | "
            f"Epsilon: {epsilon:<6.2f}"
        )
        # Early stopping - not improving
        if epsilon < 0.5 and np.mean(evaluate_rewards[-5:]) == reward_valid_env:
            break
        
        if not weight_reset and len(reward_all) >= 5 and np.mean(evaluate_rewards[-5:]) == 0:
            logging.info('Reseting weights')
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
            logging.info(f'--------------- Retrying {retry} ---------------')

    
    return reward_all, evaluate_rewards, test_rewards

if __name__=="__main__":
    tickers = Tickers.TICKERS_penny
    
    for ticker in tickers:
                logging.info(f'================ Training {ticker} ================')
                reward_all, evaluate_rewards, test_rewards = trining_retry_loop(ticker, newData = True)
                training_log_df = upsert_training_logs(reward_all, evaluate_rewards, test_rewards,ticker)
