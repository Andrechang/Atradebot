import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from src.atradebot import backtest
import matplotlib.pyplot as plt
import yfinance as yf



if 'stocks_choice' not in st.session_state:
    st.session_state.stocks_choice = {}

@st.cache_resource
def get_data(stocks, start_date, end_date):
    data = yf.download(stocks, start=start_date, end=end_date)
    return data

time_future = st.session_state.qa_answers['time']
init_capital = st.session_state.qa_answers['amount']


past_date = "2018-01-31" 
start_date = "2019-01-31"  
end_date = "2020-05-20" 
stocks = ['AAPL','ABBV','AMZN','MSFT','NVDA','TSLA', 'SPY', 'VUG', 'VOO']

st.write("""
# Generating Stock allocation
""")

data = get_data(stocks, past_date, end_date)

backtester = backtest.PortfolioBacktester(initial_capital=init_capital, data=data, stocks=stocks, start_date=start_date)
strategy = backtest.SimpleStrategy(start_date, end_date, data, stocks, init_capital)
backtester.run_backtest(strategy)

#TODO: generate plots for future projection: (highlight news)
idx = data.index.get_loc(start_date)
data_spy = data['Close']['SPY'][idx:]
data_my = backtester.portfolio['Total'][idx:]
plt = backtest.plot_cmp({"SPY":data_spy/data_spy[0], #normalize gains
        "MyPort":data_my/init_capital}, show=False)

st.pyplot(plt)

submitted = st.button("Learn more")
if submitted:
    switch_page("page3")