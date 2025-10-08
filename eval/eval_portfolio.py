import numpy as np
import matplotlib.pyplot as plt
import torch

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
        

    return total_reward, steps, info['portfolio_value']

    
    
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
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    
    # Plot 1: Portfolio value over time
    ax1 = axes[0, 0]
    if env.portfolio_value_history:
        ax1.plot(env.portfolio_value_history, color='darkgreen', linewidth=2)
        ax1.axhline(y=env.initial_cash, color='red', linestyle='--', alpha=0.7, label='Kapitał początkowy')
        profit_pct = (env.portfolio_value_history[-1] - env.initial_cash) / env.initial_cash * 100
        ax1.set_title(f'Wartość portfela w czasie\nProfit: {profit_pct:.2f}% Wartość końcowa: {env.portfolio_value_history[-1]:.2f}')
        ax1.set_ylabel('Wartość portfela')
        ax1.grid(True)
        ax1.legend()
    #print(len(env.portfolio_value_history))
    #print(len(env.asset_value_history[0]))
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
    
    # Plot 3: Current portfolio allocation (pie chart with percentages)
    ax3 = axes[1, 0]
    current_prices = env.close_data.iloc[env.current_step-1].values
    position_values = env.position * current_prices
    
    # Calculate total portfolio value
    total_portfolio_value = env.cash + np.sum(position_values)
    
    # Create pie chart data
    labels = []
    sizes = []
    percentages = []
    
    # Add cash
    if env.cash > 0:
        cash_pct = (env.cash / total_portfolio_value) * 100
        labels.append(f'Gotówka ({cash_pct:.1f}%)')
        sizes.append(env.cash)
        percentages.append(cash_pct)
    
    # Add asset positions
    for i, asset_name in enumerate(env.asset_names):
        if position_values[i] > 0:
            asset_pct = (position_values[i] / total_portfolio_value) * 100
            labels.append(f'{asset_name} ({asset_pct:.1f}%)')
            sizes.append(position_values[i])
            percentages.append(asset_pct)
    
    if sizes:
        wedges, texts, autotexts = ax3.pie(
            sizes, labels=None, autopct='%1.1f%%', startangle=90
        )
        ax3.legend(
            wedges, labels, title="Aktywa", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1)
        )
        ax3.set_title(f'Aktualna alokacja portfela\n(Całkowita wartość: {total_portfolio_value:.2f})')
    else:
        ax3.text(0.5, 0.5, 'Brak pozycji', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Aktualna alokacja portfela')
    
    # Plot 4: Trading activity heatmap
    ax4 = axes[1, 1]
    
    # Create trading activity matrix
    max_steps = max(len(env.states_buy[i]) + len(env.states_sell[i]) for i in range(env.n_assets))
    #print(len(env.states_sell[0]))
    #print(max_steps)
    if max_steps > 0:
        activity_matrix = np.zeros((env.n_assets, env.current_step - env.window_size))
    
        for i in range(env.n_assets):

            for j, step in enumerate(env.states_buy[i]):
                if step - env.window_size >= 0 and step - env.window_size < activity_matrix.shape[1]:
                    #print(step,env.asset_percentage_buy_history[i][step])
                    activity_matrix[i, step - env.window_size] = env.asset_percentage_buy_history[i][j]
                    #print(j,step,env.asset_percentage_buy_history[i][j])
            for j,step in enumerate(env.states_sell[i]):
                if step - env.window_size >= 0 and step - env.window_size < activity_matrix.shape[1]:
                    activity_matrix[i, step - env.window_size] = env.asset_percentage_sell_history[i][j] * -1  
        
        im = ax4.imshow(activity_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.3, vmax=0.3)
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
    
    # Plot 5: Portfolio allocation over time (absolute values using shares)
    ax5 = axes[2, 0]
    if hasattr(env, 'asset_value_history') and len(env.asset_value_history[0]) > 0:
        time_steps = range(len(env.asset_value_history[0]))
        
        # Calculate cash history if not available
        if hasattr(env, 'cash_history'):
            cash_history = env.cash_history
        else:
            # Reconstruct cash history from portfolio value and asset values
            cash_history = []
            for step in range(len(time_steps)):
                total_asset_value = sum(env.asset_value_history[i][step] for i in range(env.n_assets))
                if step < len(env.portfolio_value_history):
                    cash_at_step = env.portfolio_value_history[step] - total_asset_value
                    cash_history.append(max(0, cash_at_step))
                else:
                    cash_history.append(env.cash)
        
        # Plot cash
        ax5.fill_between(time_steps, 0, cash_history[:len(time_steps)], alpha=0.7, label='Gotówka', color='green')
        
        # Plot assets stacked
        bottom = cash_history[:len(time_steps)]
        colors = plt.cm.Set3(np.linspace(0, 1, env.n_assets))
        
        for i, asset_name in enumerate(env.asset_names):
            asset_values = env.asset_value_history[i][:len(time_steps)]
            ax5.fill_between(time_steps, bottom, 
                            [b + v for b, v in zip(bottom, asset_values)], 
                            alpha=0.7, label=asset_name, color=colors[i])
            bottom = [b + v for b, v in zip(bottom, asset_values)]
        
        ax5.set_title('Alokacja portfela w czasie (wartości)')
        ax5.set_xlabel('Krok czasowy')
        ax5.set_ylabel('Wartość')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Brak historii wartości aktywów', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Alokacja portfela w czasie')
    
    # Plot 6: Portfolio allocation over time (percentages)
    ax6 = axes[2, 1]
    if hasattr(env, 'asset_value_history') and len(env.asset_value_history[0]) > 0:
        time_steps = range(len(env.asset_value_history[0]))
        
        # Calculate percentage allocation history
        cash_history = []
        for step in range(len(time_steps)):
            total_asset_value = sum(env.asset_value_history[i][step] for i in range(env.n_assets))
            if step < len(env.portfolio_value_history):
                cash_at_step = env.portfolio_value_history[step] - total_asset_value
                cash_history.append(max(0, cash_at_step))
            else:
                cash_history.append(env.cash)
        
        percentage_history = {'cash': [], 'assets': {name: [] for name in env.asset_names}}
        
        for step in range(len(time_steps)):
            total_value = cash_history[step] + sum(env.asset_value_history[i][step] for i in range(env.n_assets))
            
            if total_value > 0:
                percentage_history['cash'].append((cash_history[step] / total_value) * 100)
                for i, asset_name in enumerate(env.asset_names):
                    percentage_history['assets'][asset_name].append((env.asset_value_history[i][step] / total_value) * 100)
            else:
                percentage_history['cash'].append(0)
                for asset_name in env.asset_names:
                    percentage_history['assets'][asset_name].append(0)
        
        # Plot cash percentage
        ax6.fill_between(time_steps, 0, percentage_history['cash'], 
                         alpha=0.7, label='Gotówka', color='green')
        
        # Plot assets stacked
        bottom_pct = percentage_history['cash'].copy()
        colors = plt.cm.Set3(np.linspace(0, 1, env.n_assets))
        
        for i, asset_name in enumerate(env.asset_names):
            asset_pct = percentage_history['assets'][asset_name]
            ax6.fill_between(time_steps, bottom_pct, 
                            [b + v for b, v in zip(bottom_pct, asset_pct)], 
                            alpha=0.7, label=asset_name, color=colors[i])
            bottom_pct = [b + v for b, v in zip(bottom_pct, asset_pct)]
        
        ax6.set_title('Alokacja portfela w czasie (%)')
        ax6.set_xlabel('Krok czasowy')
        ax6.set_ylabel('Procent portfela (%)')
        ax6.set_ylim(0, 100)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Brak historii wartości aktywów', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Alokacja portfela w czasie (%)')
    
    plt.suptitle(f'Podsumowanie portfela {title_suffix}', fontsize=16)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)