import streamlit as st
from eval.eval_models import evaluate_steps_for_UI
from agents.traderModel import get_trading_desk
from eval.eval_portfolio import evaluate_porfolio_steps_for_UI
from tickers import Tickers
# =============================
# Konfiguracja strony
# =============================
st.set_page_config(page_title="AI Trading Dashboard", layout="wide", page_icon="💹")

st.title("💹 AI Trading Dashboard")
st.markdown("### Monitorowanie wyników agentów i portfela")

# =============================
# Dane wejściowe
# =============================
tickers = Tickers()

# Możesz tu dodać np. rozwijane menu z kategoriami aktywów
ticker_list = tickers.TICKERS_penny

# =============================
# Menu boczne
# =============================
st.sidebar.header("📂 Wybierz widok")
view_option = st.sidebar.radio(
    "Tryb widoku:",
    ["📊 Portfolio", "🤖 Traderzy indywidualni"]
)

st.sidebar.markdown("---")
st.sidebar.write("💡 Wskazówka: zmień tryb widoku, aby zobaczyć wyniki dla portfela lub pojedynczych aktywów.")

# =============================
# Widok: PORTFOLIO
# =============================
if view_option == "📊 Portfolio":
    st.subheader("🧺 Podsumowanie portfela")

    with st.spinner("Obliczanie wyników portfela..."):
        trading_desk = get_trading_desk(ticker_list)
        evaluate_porfolio_steps_for_UI(trading_desk, window_size=96, device="cuda:0", OHCL=False)
    
    st.success("✅ Portfel został przetworzony.")
    st.markdown("---")

# =============================
# Widok: TRADERZY
# =============================
elif view_option == "🤖 Traderzy indywidualni":
    st.subheader("📈 Wyniki indywidualnych agentów")
    for ticker in ticker_list:
        st.markdown(f"### 🤖 Agent dla {ticker}")
        with st.spinner(f"Symulacja dla {ticker}..."):
            print(f"Evaluating for {ticker}")
            evaluate_steps_for_UI(ticker, window_size=96, device="cuda:0", OHCL=False)
        st.success("✅ Analiza zakończona pomyślnie.")



#https://ui.shadcn.com/
#https://tweakcn.com/