import streamlit as st
from eval.eval_models import evaluate_steps_for_UI
from tickers import Tickers

tickers = Tickers()
members = [attr for attr in dir(tickers) if not callable(getattr(tickers, attr)) and not attr.startswith("__")]
st.write(members)

for ticker in tickers.TICKERS_penny:
    st.header(f"Agent dla {ticker}")
    fig = evaluate_steps_for_UI(ticker, window_size=96, device="cuda:0", OHCL=False)
    st.pyplot(fig)


