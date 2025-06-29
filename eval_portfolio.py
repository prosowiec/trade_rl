import numpy as np
import matplotlib.pyplot as plt
import torch
# import matplotlib
# matplotlib.use("TkAgg")  # lub "QtAgg", jeśli masz Qt
# plt.ion()  # interaktywny tryb

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
        trader_actions = trader.get_action(env.get_price_window(), target_model = True)
        
        action_allocation_percentages = portfolio_manager.get_action_target(state_tensor)

        action = {
            'trader': trader_actions,
            'portfolio_manager': np.array([action_allocation_percentages]).flatten()
        }
        allocations.append(np.array([action_allocation_percentages]).flatten())
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
    
    for i, all in enumerate(allocations):
        print(i, all)
    #print(allocations)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Skalowanie alokacji - znajdź min i max
    alloc_values = [float(a[0]) if isinstance(a, (list, np.ndarray)) else float(a) for a in allocations]
    alloc_min = 0 #min(alloc_values) if alloc_values else 0
    alloc_max = max(alloc_values) if alloc_values else 1
    alloc_range = alloc_max - alloc_min if alloc_max != alloc_min else 1e-6

    # Funkcja skalująca alokację na rozmiar markerów (min 50, max 300)
    def scale_marker_size(alloc):
        min_size, max_size = 50, 300
        norm_alloc = (alloc - alloc_min) / alloc_range
        return min_size + (max_size - min_size) * norm_alloc

    def get_scalar_alloc(i):
        a = allocations[i]
        return float(a[0]) if isinstance(a, (list, np.ndarray)) else float(a)

    # Wykres ceny
    ax1.plot(prices, label='Cena', color='black', linewidth=1.5)

    if buy_points:
        buy_sizes = [scale_marker_size(get_scalar_alloc(i)) if i < len(allocations) else 50 for i in buy_points]
        ax1.scatter(buy_points, prices[buy_points], color='green', marker='^', s=buy_sizes, label='Kup')
    if sell_points:
        sell_sizes = [scale_marker_size(get_scalar_alloc(i)) if i < len(allocations) else 50 for i in sell_points]
        ax1.scatter(sell_points, prices[sell_points], color='red', marker='v', s=sell_sizes, label='Sprzedaj')

    profit = np.round((env.total_porfolio - env.initial_cash) / env.initial_cash * 100, 2)
    ax1.set_ylabel('Cena aktywa')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title(f'Działania agenta {title_suffix}\nŁączny portfel: {env.total_porfolio} Otwarte pozycje {env.position} Profit {profit} %')

    # Wykres alokacji
    if allocations:
        n = len(allocations)
        alloc_filtered = np.full(n, np.nan)
        for i in buy_points + sell_points:
            if i < n:
                alloc_filtered[i] = get_scalar_alloc(i)
        ax2.bar(range(n), np.nan_to_num(alloc_filtered), color='blue', alpha=0.6)

    ax2.set_ylim(alloc_min, alloc_max)
    ax2.set_ylabel('Alokacja')
    ax2.set_xlabel('Krok czasowy')
    ax2.grid(True)

    plt.tight_layout()
    plt.show(block=False)  # 👈 nie blokuje dalszego działania programu
    plt.pause(0.5)       # 👈 aktualizuje rysunek (potrzebne przy block=False)