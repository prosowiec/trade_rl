import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def render_portfolio_summary_streamlit(env, title_suffix=""):
    """
    Render a comprehensive portfolio summary for Streamlit with separate charts.
    
    Args:
        env: PortfolioEnv environment
        title_suffix: Optional suffix for section title
    """
    st.header(f"📊 Podsumowanie portfela {title_suffix}")

    # =====================
    # 1️⃣ Wartość portfela w czasie
    # =====================
    if getattr(env, "portfolio_value_history", None):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(env.portfolio_value_history, color='darkgreen', linewidth=2)
        ax.axhline(y=env.initial_cash, color='red', linestyle='--', alpha=0.7, label='Kapitał początkowy')

        profit_pct = (env.portfolio_value_history[-1] - env.initial_cash) / env.initial_cash * 100
        ax.set_title(f'Wartość portfela w czasie\nProfit: {profit_pct:.2f}% | Wartość końcowa: {env.portfolio_value_history[-1]:.2f}')
        ax.set_ylabel('Wartość portfela')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("Brak danych o wartości portfela 📉")

    # =====================
    # 2️⃣ Znormalizowane ceny aktywów
    # =====================
    if hasattr(env, "close_data"):
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = plt.cm.Set3(np.linspace(0, 1, env.n_assets))
        for i, asset_name in enumerate(env.asset_names):
            prices = env.close_data[asset_name].values
            normalized_prices = prices / prices[0]
            ax.plot(normalized_prices, color=colors[i], label=asset_name, linewidth=1.5)
        ax.set_title('Znormalizowane ceny aktywów')
        ax.set_ylabel('Cena znormalizowana')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("Brak danych o cenach aktywów 📈")

    # =====================
    # 3️⃣ Aktualna alokacja portfela (wykres kołowy)
    # =====================
    fig, ax = plt.subplots(figsize=(6, 6))
    current_prices = env.close_data.iloc[env.current_step - 1].values
    position_values = env.position * current_prices
    total_portfolio_value = env.cash + np.sum(position_values)

    labels, sizes = [], []

    if env.cash > 0:
        labels.append(f'Gotówka ({(env.cash / total_portfolio_value) * 100:.1f}%)')
        sizes.append(env.cash)

    for i, asset_name in enumerate(env.asset_names):
        if position_values[i] > 0:
            labels.append(f'{asset_name} ({(position_values[i] / total_portfolio_value) * 100:.1f}%)')
            sizes.append(position_values[i])

    if sizes:
        wedges, _, _ = ax.pie(sizes, autopct='%1.1f%%', startangle=90)
        ax.legend(wedges, labels, title="Aktywa", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        ax.set_title(f'Aktualna alokacja portfela\n(Całkowita wartość: {total_portfolio_value:.2f})')
        st.pyplot(fig)
    else:
        st.info("Brak otwartych pozycji 💼")

    # =====================
    # 4️⃣ Aktywność handlowa (heatmapa)
    # =====================
    fig, ax = plt.subplots(figsize=(10, 4))
    max_steps = max(len(env.states_buy[i]) + len(env.states_sell[i]) for i in range(env.n_assets))
    if max_steps > 0:
        activity_matrix = np.zeros((env.n_assets, env.current_step - env.window_size))

        for i in range(env.n_assets):
            for j, step in enumerate(env.states_buy[i]):
                if 0 <= step - env.window_size < activity_matrix.shape[1]:
                    activity_matrix[i, step - env.window_size] = env.asset_percentage_buy_history[i][j]
            for j, step in enumerate(env.states_sell[i]):
                if 0 <= step - env.window_size < activity_matrix.shape[1]:
                    activity_matrix[i, step - env.window_size] = -env.asset_percentage_sell_history[i][j]

        im = ax.imshow(activity_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.3, vmax=0.3)
        ax.set_yticks(range(env.n_assets))
        ax.set_yticklabels(env.asset_names)
        ax.set_title('Aktywność handlowa (Zielony=Kup, Czerwony=Sprzedaj)')
        ax.set_xlabel('Krok czasowy')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Akcja')
        st.pyplot(fig)
    else:
        st.info("Brak aktywności handlowej 🔇")

    # =====================
    # 5️⃣ Alokacja portfela w czasie (wartości)
    # =====================
    if hasattr(env, 'asset_value_history') and len(env.asset_value_history[0]) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        time_steps = range(len(env.asset_value_history[0]))

        if hasattr(env, 'cash_history'):
            cash_history = env.cash_history
        else:
            cash_history = []
            for step in time_steps:
                total_asset_value = sum(env.asset_value_history[i][step] for i in range(env.n_assets))
                if step < len(env.portfolio_value_history):
                    cash_at_step = env.portfolio_value_history[step] - total_asset_value
                    cash_history.append(max(0, cash_at_step))
                else:
                    cash_history.append(env.cash)

        ax.fill_between(time_steps, 0, cash_history[:len(time_steps)], alpha=0.7, label='Gotówka', color='green')
        bottom = cash_history[:len(time_steps)]
        colors = plt.cm.Set3(np.linspace(0, 1, env.n_assets))

        for i, asset_name in enumerate(env.asset_names):
            asset_values = env.asset_value_history[i][:len(time_steps)]
            ax.fill_between(time_steps, bottom, 
                            [b + v for b, v in zip(bottom, asset_values)], 
                            alpha=0.7, label=asset_name, color=colors[i])
            bottom = [b + v for b, v in zip(bottom, asset_values)]

        ax.set_title('Alokacja portfela w czasie (wartości)')
        ax.set_xlabel('Krok czasowy')
        ax.set_ylabel('Wartość')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Brak historii wartości aktywów 📊")

    # =====================
    # 6️⃣ Alokacja portfela w czasie (%)
    # =====================
    if hasattr(env, 'asset_value_history') and len(env.asset_value_history[0]) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        time_steps = range(len(env.asset_value_history[0]))

        cash_history = []
        for step in time_steps:
            total_asset_value = sum(env.asset_value_history[i][step] for i in range(env.n_assets))
            if step < len(env.portfolio_value_history):
                cash_at_step = env.portfolio_value_history[step] - total_asset_value
                cash_history.append(max(0, cash_at_step))
            else:
                cash_history.append(env.cash)

        percentage_history = {'cash': [], 'assets': {name: [] for name in env.asset_names}}

        for step in time_steps:
            total_value = cash_history[step] + sum(env.asset_value_history[i][step] for i in range(env.n_assets))
            if total_value > 0:
                percentage_history['cash'].append((cash_history[step] / total_value) * 100)
                for i, asset_name in enumerate(env.asset_names):
                    percentage_history['assets'][asset_name].append((env.asset_value_history[i][step] / total_value) * 100)
            else:
                percentage_history['cash'].append(0)
                for asset_name in env.asset_names:
                    percentage_history['assets'][asset_name].append(0)

        ax.fill_between(time_steps, 0, percentage_history['cash'], alpha=0.7, label='Gotówka', color='green')
        bottom_pct = percentage_history['cash'].copy()
        colors = plt.cm.Set3(np.linspace(0, 1, env.n_assets))

        for i, asset_name in enumerate(env.asset_names):
            asset_pct = percentage_history['assets'][asset_name]
            ax.fill_between(time_steps, bottom_pct, 
                            [b + v for b, v in zip(bottom_pct, asset_pct)], 
                            alpha=0.7, label=asset_name, color=colors[i])
            bottom_pct = [b + v for b, v in zip(bottom_pct, asset_pct)]

        ax.set_title('Alokacja portfela w czasie (%)')
        ax.set_xlabel('Krok czasowy')
        ax.set_ylabel('Procent portfela (%)')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Brak historii procentowej alokacji 📈")

def render_env_streamlit(env, title_suffix="", OHCL=False):
    if OHCL:
        prices = env.ohlc_data[:, 3]  # kolumna 'Close'
    else:
        prices = env.data

    buy_points = [i for i in env.states_buy if i < len(prices)]
    sell_points = [i for i in env.states_sell if i < len(prices)]
    profit = getattr(env, "total_profit", 0)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(prices, label='Cena', linewidth=1.5)

    if buy_points:
        ax.scatter(buy_points, [prices[i] for i in buy_points],
                   color='green', marker='^', label='Kup', s=100)
    if sell_points:
        ax.scatter(sell_points, [prices[i] for i in sell_points],
                   color='red', marker='v', label='Sprzedaj', s=100)

    ax.set_title(f'Działania agenta {title_suffix} | Łączny zysk: {profit:.2f}')
    ax.axvline(x=48, color='red', linestyle='--', label='Początek okna czasowego')
    ax.set_xlabel('Krok')
    ax.set_ylabel('Cena')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return fig