import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def evaluate_steps_portfolio(env, trader, portfolio_manager, device="cuda:0"):
    """
    Evaluate the portfolio environment with separate trader and portfolio manager models
    
    Args:
        env: PortfolioEnv environment
        trader_model: Model that outputs trading decisions (buy/sell/hold)
        portfolio_model: Model that outputs allocation percentages
        device: Device to run models on
    """
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    allocations = []
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # [1, obs_size]
        trader_actions = trader.get_action(state[:96], target_model = True)
        with torch.no_grad():
            portfolio_allocations = portfolio_manager.get_action_target(state_tensor)  # [1, n_assets]
            #portfolio_allocations = torch.sigmoid(portfolio_allocations)  # Ensure 0-1 range
            #portfolio_allocations = portfolio_allocations.squeeze(0).cpu().numpy()  # [n_assets]
        #print(portfolio_allocations)
        # Combine actions
        action = {
            'trader': trader_actions,
            'portfolio_manager': portfolio_allocations
        }
        allocations.append(portfolio_allocations)
        # Take step
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Optional: print progress for debugging
        #if steps % 100 == 0:
        #    print(f"Step {steps}, Reward: {reward}, Portfolio Value: {info['portfolio_value']}")
    #print(allocations)
    return total_reward, steps, info['portfolio_value']


def render_env(env, title_suffix=""):
    prices = env.close_data.values
    buy_points = env.states_buy
    sell_points = env.states_sell
    allocations = env.states_allocation

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Funkcja skalująca alokację na rozmiar markerów (min 50, max 300)
    def scale_marker_size(alloc):
        min_size, max_size = 50, 300
        return min_size + (max_size - min_size) * alloc

    # Pomocniczna funkcja, która wyciąga skalarną wartość z alokacji
    def get_scalar_alloc(i):
        a = allocations[i]
        # Jeśli a jest tablicą lub listą, weź pierwszy element
        if isinstance(a, (list, np.ndarray)):
            return float(a[0])
        else:
            return float(a)

    # Wykres ceny
    ax1.plot(prices, label='Cena', color='black', linewidth=1.5)

    if buy_points:
        buy_sizes = [scale_marker_size(get_scalar_alloc(i)) if i < len(allocations) else 50 for i in buy_points]
        ax1.scatter(buy_points, prices[buy_points], color='green', marker='^', s=buy_sizes, label='Kup')
    if sell_points:
        sell_sizes = [scale_marker_size(get_scalar_alloc(i)) if i < len(allocations) else 50 for i in sell_points]
        ax1.scatter(sell_points, prices[sell_points], color='red', marker='v', s=sell_sizes, label='Sprzedaj')

    ax1.set_ylabel('Cena aktywa')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title(f'Działania agenta {title_suffix}\nŁączny portfel: {env.total_porfolio} PLN')

    # Wykres alokacji - pokazujemy tylko w punktach transakcji
    if allocations:
        n = len(allocations)
        alloc_filtered = np.full(n, np.nan)
        for i in buy_points + sell_points:
            if i < n:
                alloc_filtered[i] = get_scalar_alloc(i)
        ax2.bar(range(n), np.nan_to_num(alloc_filtered), color='blue', alpha=0.6)

    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Alokacja')
    ax2.set_xlabel('Krok czasowy')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
