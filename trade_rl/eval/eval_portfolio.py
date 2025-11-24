import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.database import read_stock_data
import pandas as pd
from agent_env.manager_env import PortfolioEnv
from agents.managerModel import AgentPortfolio
from dashboardViews.streamlit_graphs import render_portfolio_summary_streamlit

def evaluate_steps_portfolio(env:PortfolioEnv, trading_desk, portfolio_manager:AgentPortfolio):
    """
    Evaluate the portfolio environment with separate trader and portfolio manager models
    
    Args:
        env: PortfolioEnv environment
        trader_model: Model that outputs trading decisions (buy/sell/hold)
        portfolio_model: Model that outputs allocation percentages
        device: Device to run models on
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    allocations = []
    portfolio_manager.mode_eval()
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        
        action_allocation_percentages = portfolio_manager.get_action_target(state_tensor)
        traders_actions = []
        hist_data = env.get_price_window()
        for i,key in enumerate(trading_desk.keys()):
            curr_trader = trading_desk[key]
            traders_actions.append(curr_trader.get_action(hist_data[i,:], target_model = True))

        action = {
            'trader' : traders_actions,
            'portfolio_manager': np.array([action_allocation_percentages]).flatten()
        }
        print(action_allocation_percentages)
        allocations.append(np.array([action_allocation_percentages]).flatten())
        # Take step
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        

    return total_reward, steps, info['portfolio_value']

def evaluate_porfolio_steps_for_UI(trading_desk : dict, window_size = 96):
    tickers = list(trading_desk.keys())
    min_size = 9999999
    data = pd.DataFrame()
    for ticker in tickers:
        train_data, valid_data, test_data = read_stock_data(ticker)
        training_set = pd.concat([train_data, valid_data, test_data])

        min_size = min(min_size, len(training_set))

        temp = pd.DataFrame(training_set['close'].copy()).rename(columns={'close': ticker})

        temp = temp[:min_size].reset_index(drop=True)
        data = data[:min_size].reset_index(drop=True)    
        data = pd.concat([data[:min_size], temp[:min_size]], axis=1)
        
    portfolio_manager = AgentPortfolio(tickers, input_dim=window_size, action_dim=len(tickers))
    portfolio_manager.load_agent()
    portfolio_manager.mode_eval()
    data_split = int(len(data)  * 0.8)
    train_data = data[:data_split]
    valid_data = data[data_split:]

    WINDOW_SIZE = 96

    valid_env = PortfolioEnv(valid_data,window_size=WINDOW_SIZE, max_allocation=.5)
    evaluate_steps_portfolio(valid_env, trading_desk, portfolio_manager)
    render_portfolio_summary_streamlit(valid_env, title_suffix="(Zbiór Walidacyjny)")

    
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