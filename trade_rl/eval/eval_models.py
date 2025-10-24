import matplotlib.pyplot as plt
import torch
import numpy as np
from agent_env.enviroments import TimeSeriesEnv_simple
from utils.database import read_stock_data
from agents.traderModel import DQNAgent
from dashboardViews.streamlit_graphs import render_env_streamlit

def evaluate_steps(env, model, OHCL = False):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    action = 0
    while not done:

        if not OHCL:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)


        with torch.no_grad():
            if OHCL:
                #state_tensor, position_ratio = state_tensor
                state = (state[0], np.full_like(state[1], 0.5))
                action = model.get_action_target(state )
            else:
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1

    return total_reward


def render_env(env, title_suffix="", OHCL = False):
    if OHCL:
        prices = env.ohlc_data[:,3]
    else:
        prices = env.data
        
    buy_points = env.states_buy
    sell_points = env.states_sell
    buy_points = [i for i in env.states_buy if i < len(prices)]
    sell_points = [i for i in env.states_sell if i < len(prices)]
    profit = env.total_profit #+ np.sum(test_env.data[-1] - test_env.inventory) 
    
    plt.figure(figsize=(14, 6))
    plt.plot(prices, label='Cena', linewidth=1.5)

    if buy_points:
        plt.scatter(buy_points, [prices[i] for i in buy_points],
                    color='green', marker='^', label='Kup', s=100)
    if sell_points:
        plt.scatter(sell_points, [prices[i] for i in sell_points],
                    color='red', marker='v', label='Sprzedaj', s=100)
    
    
    plt.title(f'Działania agenta {title_suffix} | Łączny zysk: {profit:.2f}')
    plt.axvline(x = 48, color = 'red', label = 'Początek okna czasowego')
    plt.xlabel('Krok')
    plt.ylabel('Cena')
        
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)
    

def render_training(training_log_df):
    plt.figure(figsize=(10, 5))
    
    plt.plot(training_log_df['trainRewards'], label='Train Reward')
    plt.plot(training_log_df['evaluateRewards'], label='Validation Reward')
    plt.plot(training_log_df['testRewards'], label='Test Reward')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards over Episodes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def evaluate_steps_for_UI(ticker, window_size = 96,  OHCL = False):
    train_data, valid_data, test_data = read_stock_data(ticker)
    
    test_env = TimeSeriesEnv_simple(test_data['close'].values, window_size=window_size)

    trader_model = DQNAgent(ticker)
    trader_model.load_dqn_agent()

    total_reward = evaluate_steps(test_env, trader_model.target_model, OHCL = False)

    render_env_streamlit(test_env, title_suffix=f"({ticker})", OHCL=OHCL)
    

