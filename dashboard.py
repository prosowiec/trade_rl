import streamlit as st
from trade_rl.eval.eval_models import evaluate_steps_for_UI
from trade_rl.agents.traderModel import get_trading_desk
from trade_rl.eval.eval_portfolio import evaluate_porfolio_steps_for_UI
from trade_rl.utils.database import load_trades_from_db
from trade_rl.utils.IB_connector import retrieve_account_and_portfolio, IBapi
import pandas as pd
import plotly.express as px
from trade_rl.tickers import Tickers
import threading

def get_portfolio_info(host="ib-gateway", port=4004, client_id=1):
    app = IBapi()
    app.connect(host, port, client_id)

    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()

    portfolio_df, account_series = retrieve_account_and_portfolio(app)

    app.disconnect()
    return portfolio_df, account_series

st.set_page_config(page_title="AI Trading Dashboard", layout="wide", page_icon="💹")

st.title("💹 AI Trading Dashboard")
st.markdown("### Monitorowanie wyników agentów i portfela")


tickers = Tickers()

# Możesz tu dodać np. rozwijane menu z kategoriami aktywów
ticker_list = tickers.TICKERS_penny


st.sidebar.header("📂 Wybierz widok")
view_option = st.sidebar.radio(
    "Tryb widoku:",
    ["📅 Historia transakcji","📊 Portfolio", "🤖 Traderzy indywidualni",]
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
    portfolio_df, account_series = get_portfolio_info()
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
        
        
#https://ui.shadcn.com/
#https://tweakcn.com/