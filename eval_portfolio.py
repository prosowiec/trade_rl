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
    shares_buy = env.shares_buy  # dodane: liczba akcji kupionych w danym kroku
    shares_sell = env.shares_sell
    
    # buy_points = [i + 96 for i in env.states_buy]
    # sell_points = [i + 96 for i in env.states_sell]
    
    print("Alokacje:", env.states_allocation)
    print("Kupione akcje:", env.shares_buy)
    print("Sprzedane akcje:", env.shares_sell)
    print("Pozycje",env.states_position)
    print("cash_state",env.cash_state)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Wykres ceny
    ax1.plot(prices, label='Cena', color='black', linewidth=1.5)

    # Skalowanie rozmiaru markerów wg liczby akcji (dla kupna)
    if shares_buy:
        shares_min = min(shares_buy)
        shares_max = max(shares_buy)
    else:
        shares_min, shares_max = 0, 1
    shares_range = shares_max - shares_min if shares_max != shares_min else 1e-6

    def scale_marker_size_by_shares(shares):
        min_size, max_size = 50, 300
        norm = (shares - shares_min) / shares_range
        return min_size + (max_size - min_size) * norm

    if buy_points:
        buy_sizes = [
            scale_marker_size_by_shares(shares_buy[j]) if j < len(shares_buy) else 50
            for j in range(len(buy_points))
        ]
        ax1.scatter(buy_points, prices[buy_points], color='green', marker='^', s=buy_sizes, label='Kup')

    if sell_points:
        # Dla uproszczenia: stały rozmiar markerów sprzedaży
        sell_sizes = [
            scale_marker_size_by_shares(shares_sell[j]) if j < len(shares_sell) else 50
            for j in range(len(shares_sell))
        ]

        ax1.scatter(sell_points, prices[sell_points], color='red', marker='v', s=sell_sizes, label='Sprzedaj')

    # Tytuł z informacjami o wyniku agenta
    profit = np.round((env.total_porfolio - env.initial_cash) / env.initial_cash * 100, 2)
    ax1.set_ylabel('Cena aktywa')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title(f'Działania agenta {title_suffix}\nŁączny portfel: {env.total_porfolio} '
                  f'Otwarte pozycje {env.position} Profit {profit} %')

    # Wykres alokacji
    if allocations:
        def get_scalar_alloc(i):
            a = allocations[i]
            return float(a[0]) if isinstance(a, (list, np.ndarray)) else float(a)

        alloc_values = [get_scalar_alloc(i) for i in range(len(allocations))]
        alloc_min = 0
        alloc_max = max(alloc_values) if alloc_values else 1

        alloc_filtered = np.full(len(allocations), np.nan)
        for i in buy_points + sell_points:
            if i < len(allocations):
                alloc_filtered[i] = get_scalar_alloc(i)
        ax2.bar(range(len(allocations)), np.nan_to_num(alloc_filtered), color='blue', alpha=0.6)

        ax2.set_ylim(alloc_min, alloc_max)

    ax2.set_ylabel('Alokacja')
    ax2.set_xlabel('Krok czasowy')
    ax2.grid(True)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)