import streamlit as st
import pandas as pd
from utils.database import load_trades_from_db
from utils.IB_connector import get_portfolio_info
import plotly.express as px

def transactions_view():
    st.subheader("📅 Historia transakcji (Timeline)")
    if st.button("🔄 Odśwież dane"):
        st.cache_data.clear()
        st.toast("Dane zostały odświeżone!", icon="🔁")
    
    trades = load_trades_from_db()
    portfolio_df, account_series, positions = get_portfolio_info()
    st.write(positions)

    if not portfolio_df.empty:
        st.subheader("📈 Portfolio:")
        st.dataframe(
            portfolio_df[[
                "symbol", "position", "avgCost", "marketPrice",
                "marketValue", "unrealizedPNL", "realizedPNL"
            ]].sort_values("marketValue", ascending=False),
            use_container_width=True,
        )
        st.write(f"Łączna wartość portfela: ${account_series['NetLiquidation']}")
        # --- WYKRES KOŁOWY ---
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
            
        except Exception as e:
            st.error(f"Błąd przy tworzeniu wykresu: {e}")
            
        if not account_series.empty:
            st.subheader("💰 Wartości konta:")
            st.write(account_series)

    if not trades:
        st.warning("Brak zapisanych transakcji w bazie danych.")
    else:
        # Tworzymy DataFrame z transakcji
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

        st.dataframe(
            matrix.style.applymap(color_action),
            use_container_width=True,
            height=500
        )

        st.markdown("🟩 BUY &nbsp;&nbsp; 🟥 SELL &nbsp;&nbsp; 🟨 HOLD")
        st.markdown("---")
        