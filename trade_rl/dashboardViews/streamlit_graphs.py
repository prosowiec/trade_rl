import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from dashboardViews.graphConfig import LEGEND,FIGURE_SHOW_COFIG

def render_portfolio_summary_streamlit(env, title_suffix=""):
    """
    Render an interactive portfolio summary for Streamlit with 2-column layout.
    
    Args:
        env: PortfolioEnv environment
        title_suffix: Optional suffix for section title
    """
    st.header(f"📊 Podsumowanie portfela {title_suffix}")

    sell = sum([len(asset) for asset in env.states_sell])
    st.write("Liczba wszystkich kroków sell:", sell)

    buy = sum([len(asset) for asset in env.states_buy])
    st.write("Liczba wszystkich kroków buy:", buy)
    
    # =====================
    # Row 1: Portfolio Value & Normalized Prices
    # =====================
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Wartość portfela w czasie")
        if getattr(env, "portfolio_value_history", None):
            profit_pct = (env.portfolio_value_history[-1] - env.initial_cash) / env.initial_cash * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=env.portfolio_value_history,
                mode='lines',
                name='Wartość portfela',
                line=dict(color='darkgreen', width=2),
                hovertemplate='Krok: %{x}<br>Wartość: %{y:.2f}<extra></extra>'
            ))
            fig.add_hline(
                y=env.initial_cash,
                line_dash="dash",
                line_color="red",
                annotation_text="Kapitał początkowy",
                annotation_position="right"
            )
            
            fig.update_layout(
                title=f'Profit: {profit_pct:.2f}% | Wartość końcowa: {env.portfolio_value_history[-1]:.2f}',
                yaxis_title='Wartość portfela',
                hovermode='x unified',
                showlegend=True,
                template='plotly_white',
                legend=LEGEND,
                xaxis=dict(
                    title_font=dict(color='black'),
                    tickfont=dict(color='black')
                ),
                yaxis=dict(
                    title_font=dict(color='black'),
                    tickfont=dict(color='black')
                )
            ) 
            
            
            st.plotly_chart(fig, use_container_width=True, config=FIGURE_SHOW_COFIG)
        else:
            st.info("Brak danych o wartości portfela 📉")

    with col2:
        st.subheader("📈 Znormalizowane ceny aktywów")
        if hasattr(env, "close_data"):
            fig = go.Figure()
            colors = px.colors.qualitative.Set3
            
            for i, asset_name in enumerate(env.asset_names):
                prices = env.close_data[asset_name].values
                normalized_prices = prices / prices[0]
                fig.add_trace(go.Scatter(
                    y=normalized_prices,
                    mode='lines',
                    name=asset_name,
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f'{asset_name}<br>Krok: %{{x}}<br>Cena: %{{y:.4f}}<extra></extra>'
                ))
            
            fig.update_layout(
                yaxis_title='Cena znormalizowana',
                hovermode='x unified',
                height=400,
                showlegend=True,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                xaxis=dict(
                    title_font=dict(color='black'),
                    tickfont=dict(color='black')
                ),
                yaxis=dict(
                    title_font=dict(color='black'),
                    tickfont=dict(color='black')
                )
            )
            st.plotly_chart(fig, use_container_width=True, config=FIGURE_SHOW_COFIG)
        else:
            st.info("Brak danych o cenach aktywów 📈")

    # =====================
    # Row 2: Current Allocation & Trading Activity
    # =====================
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🥧 Aktualna alokacja portfela")
        current_prices = env.close_data.iloc[env.current_step - 1].values
        position_values = env.position * current_prices
        total_portfolio_value = env.cash + np.sum(position_values)

        labels, values = [], []

        if env.cash > 0:
            labels.append(f'Gotówka')
            values.append(env.cash)

        for i, asset_name in enumerate(env.asset_names):
            if position_values[i] > 0:
                labels.append(asset_name)
                values.append(position_values[i])

        if values:
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hovertemplate='%{label}<br>Wartość: %{value:.2f}<br>Procent: %{percent}<extra></extra>',
                textposition='auto',
                textinfo='label+percent'
            )])
            
            fig.update_layout(
                title=f'Całkowita wartość: {total_portfolio_value:.2f}',
                height=400,
                showlegend=True,
                legend=LEGEND,
                xaxis=dict(
                    title_font=dict(color='black'),
                    tickfont=dict(color='black')
                ),
                yaxis=dict(
                    title_font=dict(color='black'),
                    tickfont=dict(color='black')
                )
            )
            st.plotly_chart(fig, use_container_width=True, config=FIGURE_SHOW_COFIG)
        else:
            st.info("Brak otwartych pozycji 💼")

    with col2:
        st.subheader("🔥 Aktywność handlowa")
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

            fig = go.Figure(data=go.Heatmap(
                z=activity_matrix,
                y=env.asset_names,
                colorscale='RdYlGn',
                zmid=0,
                zmin=-0.3,
                zmax=0.3,
                hovertemplate='Aktywo: %{y}<br>Krok: %{x}<br>Akcja: %{z:.3f}<extra></extra>',
                colorbar=dict(title="Akcja")
            ))
            
            fig.update_layout(
                title='Zielony=Kup, Czerwony=Sprzedaj',
                xaxis_title='Krok czasowy',
                legend=LEGEND,
                xaxis=dict(
                    title_font=dict(color='black'),
                    tickfont=dict(color='black')
                ),
                yaxis=dict(
                    title_font=dict(color='black'),
                    tickfont=dict(color='black')
                )
            )
            st.plotly_chart(fig, use_container_width=True, config=FIGURE_SHOW_COFIG)
        else:
            st.info("Brak aktywności handlowej 🔇")

    # =====================
    # Row 3: Portfolio Allocation Over Time (Values & Percentages)
    # =====================
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Alokacja portfela - wartości")
        if hasattr(env, 'asset_value_history') and len(env.asset_value_history[0]) > 0:
            time_steps = list(range(len(env.asset_value_history[0])))

            # Calculate cash history
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

            fig = go.Figure()
            
            # Add cash
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=cash_history[:len(time_steps)],
                fill='tozeroy',
                name='Gotówka',
                line=dict(color='green'),
                hovertemplate='Krok: %{x}<br>Gotówka: %{y:.2f}<extra></extra>'
            ))
            
            # Add assets
            colors = px.colors.qualitative.Set3
            cumulative = cash_history[:len(time_steps)].copy()
            
            for i, asset_name in enumerate(env.asset_names):
                asset_values = env.asset_value_history[i][:len(time_steps)]
                next_cumulative = [c + v for c, v in zip(cumulative, asset_values)]
                
                fig.add_trace(go.Scatter(
                    x=time_steps,
                    y=next_cumulative,
                    fill='tonexty',
                    name=asset_name,
                    line=dict(color=colors[i % len(colors)]),
                    hovertemplate=f'{asset_name}<br>Krok: %{{x}}<br>Wartość: %{{y:.2f}}<extra></extra>'
                ))
                
                cumulative = next_cumulative

            fig.update_layout(
                xaxis_title='Krok czasowy',
                yaxis_title='Wartość',
                hovermode='x unified',
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True, config=FIGURE_SHOW_COFIG)
        else:
            st.info("Brak historii wartości aktywów 📊")

    with col2:
        st.subheader("📈 Alokacja portfela - procenty")
        if hasattr(env, 'asset_value_history') and len(env.asset_value_history[0]) > 0:
            time_steps = list(range(len(env.asset_value_history[0])))

            # Calculate cash history
            cash_history = []
            for step in time_steps:
                total_asset_value = sum(env.asset_value_history[i][step] for i in range(env.n_assets))
                if step < len(env.portfolio_value_history):
                    cash_at_step = env.portfolio_value_history[step] - total_asset_value
                    cash_history.append(max(0, cash_at_step))
                else:
                    cash_history.append(env.cash)

            # Calculate percentages
            percentage_data = {'cash': []}
            for asset_name in env.asset_names:
                percentage_data[asset_name] = []

            for step in time_steps:
                total_value = cash_history[step] + sum(env.asset_value_history[i][step] for i in range(env.n_assets))
                if total_value > 0:
                    percentage_data['cash'].append((cash_history[step] / total_value) * 100)
                    for i, asset_name in enumerate(env.asset_names):
                        percentage_data[asset_name].append((env.asset_value_history[i][step] / total_value) * 100)
                else:
                    percentage_data['cash'].append(0)
                    for asset_name in env.asset_names:
                        percentage_data[asset_name].append(0)

            fig = go.Figure()
            
            # Add cash
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=percentage_data['cash'],
                fill='tozeroy',
                name='Gotówka',
                line=dict(color='green'),
                hovertemplate='Krok: %{x}<br>Gotówka: %{y:.2f}%<extra></extra>'
            ))
            
            # Add assets
            colors = px.colors.qualitative.Set3
            cumulative_pct = percentage_data['cash'].copy()
            
            for i, asset_name in enumerate(env.asset_names):
                asset_pct = percentage_data[asset_name]
                next_cumulative = [c + p for c, p in zip(cumulative_pct, asset_pct)]
                
                fig.add_trace(go.Scatter(
                    x=time_steps,
                    y=next_cumulative,
                    fill='tonexty',
                    name=asset_name,
                    line=dict(color=colors[i % len(colors)]),
                    hovertemplate=f'{asset_name}<br>Krok: %{{x}}<br>Procent: %{{y:.2f}}%<extra></extra>'
                ))
                
                cumulative_pct = next_cumulative

            fig.update_layout(
                xaxis_title='Krok czasowy',
                yaxis_title='Procent portfela (%)',
                yaxis_range=[0, 100],
                hovermode='x unified',
                showlegend=True,
                legend=LEGEND,
                xaxis=dict(
                    title_font=dict(color='black'),
                    tickfont=dict(color='black')
                ),
                yaxis=dict(
                    title_font=dict(color='black'),
                    tickfont=dict(color='black')
                )
            )
            st.plotly_chart(fig, use_container_width=True, config=FIGURE_SHOW_COFIG)
        else:
            st.info("Brak historii procentowej alokacji 📈")

def render_env_streamlit(env, title_suffix="", OHCL=False):
    """
    Render an interactive trading environment visualization with Plotly.
    
    Args:
        env: Trading environment
        title_suffix: Optional suffix for title
        OHCL: If True, use OHLC data; otherwise use simple price data
    """
    # Extract price data
    if OHCL:
        prices = env.ohlc_data[:, 3]  # kolumna 'Close'
    else:
        prices = env.data
    
    # Get buy/sell points
    buy_points = [i for i in env.states_buy if i < len(prices)]
    sell_points = [i for i in env.states_sell if i < len(prices)]
    profit = getattr(env, "total_profit", 0)
    
    # Create figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=list(range(len(prices))),
        y=prices,
        mode='lines',
        name='Cena',
        line=dict(color='blue', width=2),
        hovertemplate='Krok: %{x}<br>Cena: %{y:.2f}<extra></extra>'
    ))
    
    # Add buy points
    if buy_points:
        buy_prices = [prices[i] for i in buy_points]
        fig.add_trace(go.Scatter(
            x=buy_points,
            y=buy_prices,
            mode='markers',
            name='Kup',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='green',
                line=dict(color='darkgreen', width=1)
            ),
            hovertemplate='Krok: %{x}<br>Kup po: %{y:.2f}<extra></extra>'
        ))
    
    # Add sell points
    if sell_points:
        sell_prices = [prices[i] for i in sell_points]
        fig.add_trace(go.Scatter(
            x=sell_points,
            y=sell_prices,
            mode='markers',
            name='Sprzedaj',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='red',
                line=dict(color='darkred', width=1)
            ),
            hovertemplate='Krok: %{x}<br>Sprzedaj po: %{y:.2f}<extra></extra>'
        ))
    
    # Add vertical line at step 48
    fig.add_vline(
        x=96,
        line_dash="dash",
        line_color="red",
        annotation_text="Początek okna czasowego",
        annotation_position="top right"
    )
    
    # Update layout
    profit_color = 'green' if profit >= 0 else 'red'
    fig.update_layout(
        title={
            'text': f'Działania agenta {title_suffix}<br><sub style="color:{profit_color}">Łączny zysk: {profit:.2f}</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Krok',
        yaxis_title='Cena',
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white',
        xaxis=dict(
            title_font=dict(color='black'),
            tickfont=dict(color='black')
        ),
        yaxis=dict(
            title_font=dict(color='black'),
            tickfont=dict(color='black')
        )
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    st.plotly_chart(fig, use_container_width=True, config=FIGURE_SHOW_COFIG)
    
    # Add statistics below the chart
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Łączny zysk", f"{profit:.2f}", delta=f"{profit:.2f}")
    
    with col2:
        st.metric("🟢 Transakcje kupna", len(buy_points))
    
    with col3:
        st.metric("🔴 Transakcje sprzedaży", len(sell_points))
    
    with col4:
        total_trades = len(buy_points) + len(sell_points)
        st.metric("📊 Łączne transakcje", total_trades)