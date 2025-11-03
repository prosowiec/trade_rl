import streamlit as st
import pandas as pd
from utils.database import load_trades_from_db
from utils.IB_connector import get_portfolio_info
from utils.dataOps import get_recent_data_for_UI
import plotly.express as px
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
def plot_trader_vs_prices(prices_df, trades_df, selected_asset=None, date_fmt="%m-%d %H:%M"):

    if prices_df.empty or trades_df.empty:
        st.warning("⚠️ Brak danych do wizualizacji.")
        return

    # --- Przygotowanie danych cenowych ---
    prices_df = prices_df.copy()
    if "Date" in prices_df.columns:
        prices_df["Date"] = pd.to_datetime(prices_df["Date"])
        prices_df = prices_df.set_index("Date")
    prices_df = prices_df.sort_index()

    # --- Przygotowanie danych transakcji ---
    trades_df = trades_df.copy()
    trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])

    # --- Lista aktywów ---
    assets = [selected_asset] if selected_asset else prices_df.columns

    for asset in assets:
        if asset not in prices_df.columns:
            st.info(f"⚠️ Brak danych dla {asset}")
            continue

        asset_prices = prices_df[asset].dropna()
        if asset_prices.empty:
            st.info(f"⚠️ Brak danych cenowych dla {asset}")
            continue

        dates_list = asset_prices.index.to_list()
        N = len(dates_list)
        x_idx = np.arange(N)

        df_plot = pd.DataFrame({
            "x_idx": x_idx,
            "Price": asset_prices.values,
            "Date": dates_list  # datetime-like
        })

        fig = go.Figure()

        # Linia ceny (x = równomierne indeksy)
        fig.add_trace(go.Scatter(
            x=df_plot["x_idx"],
            y=df_plot["Price"],
            mode="lines",
            name="Cena",
            line=dict(color="black", width=2),
            # customdata umożliwia dostęp do oryginalnej daty w hovertemplate
            customdata=df_plot["Date"].astype(str),
            hovertemplate="Data: %{customdata}<br>Cena: %{y}<extra></extra>"
        ))

        # --- Transakcje dla tego aktywa ---
        asset_trades = trades_df[trades_df["asset_name"] == asset].copy()

        if not asset_trades.empty:
            # BUY
            buy_trades = asset_trades[asset_trades["action"] == "BUY"]
            if not buy_trades.empty:
                buy_x = []
                buy_y = []
                buy_dates_for_hover = []
                for _, trade in buy_trades.iterrows():
                    trade_time = trade["timestamp"]
                    loc = asset_prices.index.get_indexer([trade_time], method='nearest')[0]
                    if loc >= 0 and loc < N:
                        buy_x.append(loc)
                        buy_y.append(asset_prices.iloc[loc])
                        buy_dates_for_hover.append(str(asset_prices.index[loc]))
                if len(buy_x) > 0:
                    fig.add_trace(go.Scatter(
                        x=buy_x,
                        y=buy_y,
                        mode="markers",
                        name="BUY",
                        marker=dict(symbol="triangle-up", size=14, line=dict(color="darkgreen", width=1.5), color="green"),
                        customdata=buy_dates_for_hover,
                        hovertemplate="BUY<br>Data: %{customdata}<br>Cena: %{y}<extra></extra>"
                    ))

            # SELL
            sell_trades = asset_trades[asset_trades["action"] == "SELL"]
            if not sell_trades.empty:
                sell_x = []
                sell_y = []
                sell_dates_for_hover = []
                for _, trade in sell_trades.iterrows():
                    trade_time = trade["timestamp"]
                    loc = asset_prices.index.get_indexer([trade_time], method='nearest')[0]
                    if loc >= 0 and loc < N:
                        sell_x.append(loc)
                        sell_y.append(asset_prices.iloc[loc])
                        sell_dates_for_hover.append(str(asset_prices.index[loc]))
                if len(sell_x) > 0:
                    fig.add_trace(go.Scatter(
                        x=sell_x,
                        y=sell_y,
                        mode="markers",
                        name="SELL",
                        marker=dict(symbol="triangle-down", size=14, line=dict(color="darkred", width=1.5), color="red"),
                        customdata=sell_dates_for_hover,
                        hovertemplate="SELL<br>Data: %{customdata}<br>Cena: %{y}<extra></extra>"
                    ))

        max_ticks = 10
        if N <= max_ticks:
            tickvals = x_idx
        else:
            tickvals = np.unique(np.linspace(0, N - 1, max_ticks, dtype=int))

        ticktext = [dates_list[int(i)].strftime(date_fmt) for i in tickvals]

        fig.update_layout(
            title=f"Działania tradera dla {asset}",
            xaxis=dict(
                tickmode="array",
                tickvals=tickvals.tolist(),
                ticktext=ticktext,
                title="Czas (z pominięciem dni zamkniętej giełdy)",
                showgrid=False
            ),
            yaxis=dict(title="Cena", showgrid=True),
            hovermode="x unified",
            template="plotly_white",
            margin=dict(l=40, r=20, t=60, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)
                                                
def transactions_view(active_tickers):
    st.subheader("📅 Historia transakcji (Timeline)")
    if st.button("🔄 Odśwież dane"):
        st.cache_data.clear()
        st.toast("Dane zostały odświeżone!", icon="🔁")
    
    trades = load_trades_from_db()
    portfolio_df, account_series, positions = get_portfolio_info()

    if not portfolio_df.empty:
        net_liquidation = float(account_series['NetLiquidation'])
        st.metric(
            label="💼 Łączna wartość portfela",
            value=f"${net_liquidation:,.2f}"
        )        
        st.subheader("🥧 Struktura portfela (według wartości rynkowej)")
        try:
            pie_df = (
                portfolio_df.groupby("symbol")["marketValue"]
                .sum()
                .reset_index()
                .sort_values("marketValue", ascending=False)
            )

            fig = px.pie(
                pie_df,
                names="symbol",
                values="marketValue",
                title="Udział aktywów w portfelu",
                hole=0.3,
            )
            fig.update_traces(textinfo="percent+label", pull=[0.05]*len(pie_df))
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("💰 Zyski i straty według tickera")
        except Exception as e:
            st.error(f"Błąd przy tworzeniu wykresu: {e}")
    try:
        pnl_df = portfolio_df.copy()
        pnl_df["totalPNL"] = pnl_df["unrealizedPNL"] + pnl_df["realizedPNL"]
        pnl_df = (
            pnl_df.groupby("symbol")["totalPNL"]
            .sum()
            .reset_index()
            .sort_values("totalPNL", ascending=False)
        )
        
        # Prepare position data
        position_df = (
            portfolio_df.groupby("symbol")["position"]
            .sum()
            .reset_index()
            .sort_values("position", ascending=False)
        )
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            # PNL bar chart
            fig_bar = px.bar(
                pnl_df,
                y="symbol",
                x="totalPNL",
                title="Łączny PNL (Unrealized + Realized)",
                color="totalPNL",
                color_continuous_scale=["red", "gray", "green"],
                labels={"totalPNL": "Total PNL ($)", "symbol": "Ticker"},
                orientation='h'
            )            
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Position bar chart
            fig_pos = px.bar(
                position_df,
                y="symbol",
                x="position",
                title="Pozycje według tickera",
                color="position",
                color_continuous_scale="Blues",
                labels={"position": "Liczba akcji", "symbol": "Ticker"},
                orientation='h'
            )
            fig_pos.update_layout(showlegend=False)
            st.plotly_chart(fig_pos, use_container_width=True)
            
    except Exception as e:
        st.error(f"Błąd przy tworzeniu wykresów: {e}")        
    if not trades:
        st.warning("Brak zapisanych transakcji w bazie danych.")
    else:
        data = []
        for t in trades:
            ts = (
                t.timestamp.strftime("%Y-%m-%d %H:%M")
                if hasattr(t.timestamp, "strftime")
                else str(t.timestamp)[:16]
            )
            data.append({
                "id": t.id,
                "timestamp": ts,
                "asset_name": t.asset_name,
                "action": t.action,
                "allocation": getattr(t, "allocation", None),
                "price": getattr(t, "price", None),
                "position": getattr(t, "position", None),
                "executed": getattr(t, "executed", None),
            })

        df = pd.DataFrame(data)

        # 🔧 KONWERSJA na datetime (tu jest klucz!)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M", errors="coerce")

        # ✅ Teraz sortowanie po dacie działa poprawnie
        df_sorted = df.sort_values(["asset_name", "timestamp"], ascending=[True, True])
        latest_actions = (
            df_sorted.groupby("asset_name")
            .tail(10)  # tail zamiast head, żeby brać najnowsze
            .reset_index(drop=True)
        )

        # Pivot table: aktywa w wierszach, timestamp w kolumnach
        matrix = latest_actions.pivot_table(
            index="asset_name", columns="timestamp", values="action", aggfunc="last"
        )
        # --- Kolorowanie macierzy ---
        def color_action(val):
            if val == "BUY":
                color = "#2ecc71"  # zielony
            elif val == "SELL":
                color = "#e74c3c"  # czerwony
            elif val == "HOLD":
                color = "#f1c40f"  # żółty
            else:
                color = "#95a5a6"  # szary
            return f"background-color: {color}; color: black; text-align:center"

        st.header("Ostatnie działania traderów")
        st.dataframe(
            matrix.style.applymap(color_action),
            use_container_width=True,
            height=500
        )

        st.markdown("🟩 BUY &nbsp;&nbsp; 🟥 SELL &nbsp;&nbsp; 🟨 HOLD")
        st.markdown("---")
        st.header("Wizualizacja działań poszczególnych agentów na przestrzeni czasu")
        st.info("Pokazane sygnały kupna/sprzedaży zostały wygenerowane tylko wtedy kiedy algorytm był włączony")
        with st.spinner("Czekam na dane..."):
            prices_df = get_recent_data_for_UI(active_tickers)
            plot_trader_vs_prices(prices_df, df, selected_asset=None)
    
    if not account_series.empty:
        st.subheader("💰 Szczegółowe wartości konta:")
        st.write(account_series)
