import numpy as np
import matplotlib.pyplot as plt
import torch

# import matplotlib
# matplotlib.use("TkAgg")  # lub "QtAgg", jeśli masz Qt
# plt.ion()  # interaktywny tryb

def evaluate_steps_portfolio(env, trading_desk, portfolio_manager, device="cuda:0"):
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
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device) #.unsqueeze(0)  # [1, obs_size]
        #trader_actions = trader.get_action(env.get_price_window(), target_model = True)
        
        action_allocation_percentages = portfolio_manager.get_action_target(state_tensor)

        traders_actions = []
        hist_data = env.get_price_window()
        for i,key in enumerate(trading_desk.keys()):
            curr_trader = trading_desk[key]
            #print(hist_data.shape)
            #print(hist_data)
            traders_actions.append(curr_trader.get_action(hist_data[i,:], target_model = True))

        #print(traders_actions)
        action = {
            #'trader': trader.get_action(env.get_price_window(), target_model = True),
            'trader' : traders_actions,
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


# def render_env(env, title_suffix="", asset_index=None):
#     """
#     Render the portfolio environment for visualization.
    
#     Args:
#         env: PortfolioEnv environment
#         title_suffix: Optional suffix for plot title
#         asset_index: Index of the asset to plot (if None, plot the first asset or aggregate)
#     """
#     # If asset_index is not specified, default to the first asset
#     if asset_index is None:
#         asset_index = 0
#     if asset_index >= env.n_assets:
#         raise ValueError(f"asset_index {asset_index} is out of range for {env.n_assets} assets")

#     # Get price data for the selected asset
#     prices = env.close_data[env.asset_names[asset_index]].values
#     buy_points = env.states_buy[asset_index]
#     sell_points = env.states_sell[asset_index]
#     allocations = env.states_allocation[asset_index]
#     shares_buy = env.shares_buy[asset_index]
#     shares_sell = env.shares_sell[asset_index]

#     # Create figure with two subplots (price and allocation)
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

#     # Plot price
#     ax1.plot(prices, label=f'Cena ({env.asset_names[asset_index]})', color='black', linewidth=1.5)

#     # Scale marker sizes based on shares
#     shares_min = min(min(shares_buy, default=0), min(shares_sell, default=0))
#     shares_max = max(max(shares_buy, default=1), max(shares_sell, default=1))
#     shares_range = shares_max - shares_min if shares_max != shares_min else 1e-6

#     def scale_marker_size_by_shares(shares):
#         min_size, max_size = 50, 300
#         norm = (shares - shares_min) / shares_range
#         return min_size + (max_size - min_size) * norm

#     # Plot buy points
#     if buy_points:
#         buy_sizes = [scale_marker_size_by_shares(shares_buy[j]) for j in range(len(buy_points))]
#         ax1.scatter(buy_points, prices[buy_points], color='green', marker='^', s=buy_sizes, label='Kup')

#     # Plot sell points
#     if sell_points:
#         sell_sizes = [scale_marker_size_by_shares(shares_sell[j]) for j in range(len(sell_points))]
#         ax1.scatter(sell_points, prices[sell_points], color='red', marker='v', s=sell_sizes, label='Sprzedaj')

#     # Set title and labels for price plot
#     profit = np.round((env.total_portfolio - env.initial_cash) / env.initial_cash * 100, 2)
#     ax1.set_ylabel('Cena aktywa')
#     ax1.legend()
#     ax1.grid(True)
#     ax1.set_title(f'Działania agenta dla {env.asset_names[asset_index]} {title_suffix}\n'
#                   f'Łączny portfel: {env.total_portfolio:.2f} '
#                   f'Otwarte pozycje: {env.position[asset_index]:.2f} '
#                   f'Profit: {profit}%')

#     # Plot allocations
#     if allocations:
#         alloc_values = [float(a) if isinstance(a, (list, np.ndarray)) else a for a in allocations]
#         alloc_filtered = np.full(len(prices), np.nan)
#         for i in buy_points + sell_points:
#             if i < len(allocations):
#                 alloc_filtered[i] = alloc_values[i]
#         ax2.bar(range(len(prices)), np.nan_to_num(alloc_filtered), color='blue', alpha=0.6)
#         ax2.set_ylim(0, max(alloc_values, default=1))

#     ax2.set_ylabel('Alokacja')
#     ax2.set_xlabel('Krok czasowy')
#     ax2.grid(True)

#     plt.tight_layout()
#     plt.show(block=False)
#     plt.pause(0.5)
    
    
def render_env(env, title_suffix="", asset_index=None):
    """
    Render the portfolio environment for visualization.
   
    Args:
        env: PortfolioEnv environment
        title_suffix: Optional suffix for plot title
        asset_index: Index of the asset to plot (if None, plot the first asset or aggregate)
    """
    # If asset_index is not specified, default to the first asset
    if asset_index is None:
        asset_index = 0
    if asset_index >= env.n_assets:
        raise ValueError(f"asset_index {asset_index} is out of range for {env.n_assets} assets")
    
    # Get price data for the selected asset
    prices = env.close_data[env.asset_names[asset_index]].values
    buy_points = env.states_buy[asset_index]
    sell_points = env.states_sell[asset_index]
    allocations = env.states_allocation[asset_index]
    shares_buy = env.shares_buy[asset_index]
    shares_sell = env.shares_sell[asset_index]
    
    # Create figure with three subplots (price, selected asset allocation, all allocations)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True, 
                                        gridspec_kw={'height_ratios': [3, 1, 2]})
    
    # Plot price
    ax1.plot(prices, label=f'Cena ({env.asset_names[asset_index]})', color='black', linewidth=1.5)
    
    # Scale marker sizes based on shares
    all_shares = []
    for i in range(env.n_assets):
        all_shares.extend(env.shares_buy[i])
        all_shares.extend(env.shares_sell[i])
    
    shares_min = min(all_shares) if all_shares else 0
    shares_max = max(all_shares) if all_shares else 1
    shares_range = shares_max - shares_min if shares_max != shares_min else 1e-6
    
    def scale_marker_size_by_shares(shares):
        min_size, max_size = 50, 300
        if shares == 0:
            return min_size
        norm = (shares - shares_min) / shares_range
        return min_size + (max_size - min_size) * norm
    
    # Plot buy points
    if buy_points:
        buy_sizes = [scale_marker_size_by_shares(shares_buy[j]) for j in range(len(buy_points))]
        ax1.scatter(buy_points, prices[buy_points], color='green', marker='^', s=buy_sizes, label='Kup')
    
    # Plot sell points
    if sell_points:
        sell_sizes = [scale_marker_size_by_shares(shares_sell[j]) for j in range(len(sell_points))]
        ax1.scatter(sell_points, prices[sell_points], color='red', marker='v', s=sell_sizes, label='Sprzedaj')
    
    # Set title and labels for price plot
    current_portfolio_value = env.cash + np.sum(env.position * env.close_data.iloc[env.current_step-1].values)
    profit = np.round((current_portfolio_value - env.initial_cash) / env.initial_cash * 100, 2)
    ax1.set_ylabel('Cena aktywa')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title(f'Działania agenta dla {env.asset_names[asset_index]} {title_suffix}\n'
                  f'Łączny portfel: {current_portfolio_value:.2f} '
                  f'Otwarte pozycje: {env.position[asset_index]:.2f} '
                  f'Profit: {profit}%')
    
    # Plot allocations for selected asset
    if allocations:
        alloc_values = [float(a) if isinstance(a, (list, np.ndarray)) else a for a in allocations]
        ax2.plot(range(len(alloc_values)), alloc_values, color='blue', linewidth=2, alpha=0.7)
        ax2.fill_between(range(len(alloc_values)), alloc_values, alpha=0.3, color='blue')
        ax2.set_ylim(0, 1)
    
    ax2.set_ylabel(f'Alokacja\n{env.asset_names[asset_index]}')
    ax2.grid(True)
    
    # Plot allocations for all assets
    max_steps = max(len(env.states_allocation[i]) for i in range(env.n_assets) if env.states_allocation[i])
    
    # Create a color map for different assets
    colors = plt.cm.Set3(np.linspace(0, 1, env.n_assets))
    
    for i in range(env.n_assets):
        if env.states_allocation[i]:
            alloc_values = [float(a) if isinstance(a, (list, np.ndarray)) else a for a in env.states_allocation[i]]
            
            # Pad with zeros if needed to match max_steps
            while len(alloc_values) < max_steps:
                alloc_values.append(0.0)
            
            ax3.plot(range(len(alloc_values)), alloc_values, 
                    color=colors[i], linewidth=2, alpha=0.8, 
                    label=env.asset_names[i])
            ax3.fill_between(range(len(alloc_values)), alloc_values, 
                           alpha=0.2, color=colors[i])
    
    ax3.set_ylabel('Alokacja wszystkich aktywów')
    ax3.set_xlabel('Krok czasowy')
    ax3.set_ylim(0, 1)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True)
    
    # Add a summary text box
    summary_text = f"Aktywa: {env.n_assets}\n"
    summary_text += f"Gotówka: {env.cash:.2f}\n"
    summary_text += f"Całkowita wartość: {current_portfolio_value:.2f}\n"
    summary_text += f"Pozycje:\n"
    for i, asset_name in enumerate(env.asset_names):
        if env.position[i] > 0:
            current_price = env.close_data.iloc[env.current_step-1, i]
            position_value = env.position[i] * current_price
            summary_text += f"  {asset_name}: {env.position[i]:.2f} ({position_value:.2f})\n"
    
    ax3.text(1.02, 0.5, summary_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)

def render_portfolio_summary(env, title_suffix=""):
    """
    Render a comprehensive portfolio summary with all assets.
    
    Args:
        env: PortfolioEnv environment
        title_suffix: Optional suffix for plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Portfolio value over time
    ax1 = axes[0, 0]
    if env.portfolio_value_history:
        ax1.plot(env.portfolio_value_history, color='darkgreen', linewidth=2)
        ax1.axhline(y=env.initial_cash, color='red', linestyle='--', alpha=0.7, label='Kapitał początkowy')
        profit_pct = (env.portfolio_value_history[-1] - env.initial_cash) / env.initial_cash * 100
        ax1.set_title(f'Wartość portfela w czasie\nProfit: {profit_pct:.2f}%')
        ax1.set_ylabel('Wartość portfela')
        ax1.grid(True)
        ax1.legend()
    
    # Plot 2: Asset prices (normalized)
    ax2 = axes[0, 1]
    colors = plt.cm.Set3(np.linspace(0, 1, env.n_assets))
    for i, asset_name in enumerate(env.asset_names):
        prices = env.close_data[asset_name].values
        normalized_prices = prices / prices[0]  # Normalize to starting price
        ax2.plot(normalized_prices, color=colors[i], label=asset_name, linewidth=1.5)
    ax2.set_title('Znormalizowane ceny aktywów')
    ax2.set_ylabel('Cena znormalizowana')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Current portfolio allocation (pie chart)
    ax3 = axes[1, 0]
    current_prices = env.close_data.iloc[env.current_step-1].values
    position_values = env.position * current_prices
    
    # Create pie chart data
    labels = []
    sizes = []
    
    # Add cash
    if env.cash > 0:
        labels.append('Gotówka')
        sizes.append(env.cash)
    
    # Add asset positions
    for i, asset_name in enumerate(env.asset_names):
        if position_values[i] > 0:
            labels.append(asset_name)
            sizes.append(position_values[i])
    
    if sizes:
        ax3.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Aktualna alokacja portfela')
    else:
        ax3.text(0.5, 0.5, 'Brak pozycji', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Aktualna alokacja portfela')
    
    # Plot 4: Trading activity heatmap
    ax4 = axes[1, 1]
    
    # Create trading activity matrix
    max_steps = max(len(env.states_buy[i]) + len(env.states_sell[i]) for i in range(env.n_assets))
    if max_steps > 0:
        activity_matrix = np.zeros((env.n_assets, env.current_step - env.window_size))
        
        for i in range(env.n_assets):
            # Mark buy actions
            for step in env.states_buy[i]:
                if step - env.window_size >= 0 and step - env.window_size < activity_matrix.shape[1]:
                    activity_matrix[i, step - env.window_size] = 1
            # Mark sell actions
            for step in env.states_sell[i]:
                if step - env.window_size >= 0 and step - env.window_size < activity_matrix.shape[1]:
                    activity_matrix[i, step - env.window_size] = -1
        
        im = ax4.imshow(activity_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
        ax4.set_yticks(range(env.n_assets))
        ax4.set_yticklabels(env.asset_names)
        ax4.set_title('Aktywność handlowa\n(Zielony=Kup, Czerwony=Sprzedaj)')
        ax4.set_xlabel('Krok czasowy')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Akcja')
    else:
        ax4.text(0.5, 0.5, 'Brak aktywności handlowej', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Aktywność handlowa')
    
    plt.suptitle(f'Podsumowanie portfela {title_suffix}', fontsize=16)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)