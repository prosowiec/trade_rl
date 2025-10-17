import streamlit as st
from eval.eval_models import evaluate_steps_for_UI
from agents.traderModel import get_trading_desk
from eval.eval_portfolio import evaluate_porfolio_steps_for_UI
from utils.database import load_trades_from_db
import pandas as pd
from tickers import Tickers

st.set_page_config(page_title="AI Trading Dashboard", layout="wide", page_icon="💹")

st.title("💹 AI Trading Dashboard")
st.markdown("### Monitorowanie wyników agentów i portfela")


tickers = Tickers()

# Możesz tu dodać np. rozwijane menu z kategoriami aktywów
ticker_list = tickers.TICKERS_penny


st.sidebar.header("📂 Wybierz widok")
view_option = st.sidebar.radio(
    "Tryb widoku:",
    ["📊 Portfolio", "🤖 Traderzy indywidualni", "📅 Historia transakcji"]
)

st.sidebar.markdown("---")
st.sidebar.write("💡 Wskazówka: wybierz widok, aby zobaczyć portfel, agentów lub historię transakcji.")

if view_option == "📊 Portfolio":
    st.subheader("🧺 Podsumowanie portfela")

    with st.spinner("Obliczanie wyników portfela..."):
        trading_desk = get_trading_desk(ticker_list)
        evaluate_porfolio_steps_for_UI(trading_desk, window_size=96)
    
    st.success("✅ Portfel został przetworzony.")
    st.markdown("---")

elif view_option == "🤖 Traderzy indywidualni":
    st.subheader("📈 Wyniki indywidualnych agentów")
    for ticker in ticker_list:
        st.markdown(f"### 🤖 Agent dla {ticker}")
        with st.spinner(f"Symulacja dla {ticker}..."):
            print(f"Evaluating for {ticker}")
            evaluate_steps_for_UI(ticker, window_size=96, OHCL=False)
        st.success("✅ Analiza zakończona pomyślnie.")


elif view_option == "📅 Historia transakcji":
    st.subheader("📅 Historia transakcji (Timeline)")
    if st.button("🔄 Odśwież dane"):
        st.cache_data.clear()
        st.toast("Dane zostały odświeżone!", icon="🔁")
        
    trades = load_trades_from_db()

    if not trades:
        st.warning("Brak zapisanych transakcji w bazie danych.")
    else:
        # Tworzymy DataFrame z transakcji
        data = []
        for t in trades:
            ts = (
                t.timestamp.strftime("%Y-%m-%d %H:%M")
                if hasattr(t.timestamp, "strftime")
                else str(t.timestamp)
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

        # Bierzemy np. 10 ostatnich decyzji dla każdego aktywa
        df_sorted = df.sort_values(["asset_name", "timestamp"], ascending=[True, True])
        latest_actions = (
            df_sorted.groupby("asset_name")
            .head(10)
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

        st.subheader("🔍 Szczegóły transakcji")

        # Interaktywny wybór transakcji
        selected = st.data_editor(
            df.sort_values("timestamp", ascending=False),
            hide_index=True,
            height=400,
            use_container_width=True,
            key="trade_selector",
        )

        # Użytkownik może zaznaczyć wiersz (np. poprzez checkbox)
        selected_rows = selected[selected.get("action").notna()]

        if not selected_rows.empty:
            selected_trade = selected_rows.iloc[0]  # bierzemy pierwszy zaznaczony
            st.markdown(f"### 💬 Szczegóły dla transakcji {selected_trade['id']}")
            st.write(f"**Aktywo:** {selected_trade['asset_name']}")
            st.write(f"**Akcja:** {selected_trade['action']}")
            st.write(f"**Cena:** {selected_trade['price']}")
            st.write(f"**Pozycja:** {selected_trade['position']}")
            st.write(f"**Alokacja:** {selected_trade['allocation']}")
            st.write(f"**Wykonano:** {selected_trade['executed']}")
            st.write(f"**Czas:** {selected_trade['timestamp']}")

        
        
#https://ui.shadcn.com/
#https://tweakcn.com/