import streamlit as st
from eval.eval_models import evaluate_steps_for_UI
from agents.traderModel import get_trading_desk
from eval.eval_portfolio import evaluate_porfolio_steps_for_UI
from utils.database import get_tickers_group, set_group_state, get_active_tickers
from utils.process_managment import find_process, stop_main, start_main, restart
from dashboardViews.transactions import transactions_view
import pandas as pd
from tickers import Tickers
from utils.state_managment import set_seed


set_seed(42)

SCRIPT_NAME = "trade_rl/main.py"
st.set_page_config(page_title="Handel SI", layout="wide", page_icon="💹")

st.title("💹 Aplikacja do automatycznego podejmowania decyzji inwestycyjnych")

groups_data = get_tickers_group()
active_tickers  = get_active_tickers()
active_group = [row["Group"] for row in groups_data if row["Active"]][0] if len(active_tickers) > 0 else None
groups = sorted(list({row["Group"] for row in groups_data}))

with st.sidebar:
    st.title("⚙️ Sterowanie aplikacją `main.py`")

    proc = find_process(SCRIPT_NAME)

    if proc:
        st.success(f"`{SCRIPT_NAME}` jest uruchomiony (PID {proc.pid}).")
        if st.button("🟥 Zatrzymaj main.py"):
            stop_main(proc)
            st.warning("Zatrzymano `main.py`.")
            st.rerun()
    else:
        st.warning(f"`{SCRIPT_NAME}` nie działa.")
        if st.button("🟩 Uruchom main.py"):
            start_main(SCRIPT_NAME)
            st.success("Uruchomiono `main.py`.")
            st.rerun()    
        

    if active_tickers:
        st.info(f"🟢 Aktualnie aktywa grupa {active_group}, aktywa: **{', '.join(active_tickers)}**")
    else:
        st.warning("⚪ Brak aktywnej grupy.")
        
    selected_group = st.selectbox("Wybierz grupę do aktywacji:", groups, index=groups.index(active_group) if active_group in groups else 0)


    if st.button("🔄 Ustaw jako aktywną grupę"):
        set_group_state(selected_group)
        st.success(f"✅ Grupa **{selected_group}** została ustawiona jako aktywna.")
        restart(SCRIPT_NAME)
        st.rerun()  # odśwież stronę po zmianie



    st.header("📂 Wybierz widok")
    view_option = st.radio(
        "Tryb widoku:",
        ["📅 Historia transakcji","📊 Wyniki testowe - agent portfolio", "🤖 Wyniki testowe - agenci handlujący"]
    )

if view_option == "📊 Wyniki testowe - agent portfolio":
    with st.spinner("Obliczanie wyników portfela..."):
        trading_desk = get_trading_desk(active_tickers)
        evaluate_porfolio_steps_for_UI(trading_desk, window_size=96)
    
    st.success("✅ Portfel został przetworzony.")
    st.markdown("---")

elif view_option == "🤖 Wyniki testowe - agenci handlujący":
    st.subheader("📈 Wyniki indywidualnych agentów")
    for ticker in active_tickers:
        st.markdown(f"### 🤖 Agent dla {ticker}")
        with st.spinner(f"Symulacja dla {ticker}..."):
            print(f"Evaluating for {ticker}")
            evaluate_steps_for_UI(ticker, window_size=96, OHCL=False)
        st.success("✅ Analiza zakończona pomyślnie.")


elif view_option == "📅 Historia transakcji":
    transactions_view(active_tickers)

        
#https://ui.shadcn.com/
#https://tweakcn.com/