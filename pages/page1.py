# SimpleStrategy: Test simple strategy
# learning points: 
#       average cost strategy reduces risk
#       invest early is better than invest late
#       diversify reduces risk
#       potential return is inversely proportional to risk: more risk (volatility == higher std), more return
#       efficient frontier: risk vs return is not linear
#       invest is not trading: invest is long term, trading is short term
#       Efficient Market Hypothesis: stocks always trade at their fair value on exchanges 
# FinForecastStrategy: Test strategy using model to forecast stock gain/loss based on news sentiment
# uses average cost together with model forecast to allocate buy/sell of selected stocks
# learning points: 
#       short term prediction (1 month) is more accurate than long term predictions (1 year)
#       train and run a model on news sentiment to forecast stock price




import streamlit as st
from atradebot import backtest, strategies, utils
import matplotlib.pyplot as plt
import yfinance as yf
import datetime

@st.cache_resource
def get_data(stocks, start_date, end_date):
    data = yf.download(stocks, start=start_date, end=end_date)
    return data

with st.form("submit"):
    init_capital = st.number_input('Amount of cash to invest: ', value=10000)

    mode = st.selectbox(
        "Which strategy to test?",
        ("SimpleStrategy", "FinForecastStrategy"),
    )

    train_start_date = st.date_input("Train data start date: ", datetime.date(2022, 9, 5))
    train_end_date = st.date_input("Train data end date: ", datetime.date(2023, 4, 4))
    st.write(f'Collect train data from {train_start_date} to {train_end_date}')

    eval_start_date = st.date_input("Evaluation data start date: ", datetime.date(2023, 4, 5))
    eval_end_date = st.date_input("Evaluation data end date: ", datetime.date(2023, 8, 7))
    st.write(f'Collect train data from {eval_start_date} to {eval_end_date}')

    stocks = st.text_input('Enter list of stocks ids separated by comma: ', 'AAPL,ABBV,AMZN,MSFT,NVDA,TSLA')
    stocks = stocks.replace(' ','')
    stocks = stocks.split(',')
    stocks += ['SPY']

    mhub = st.text_input('Enter huggingface model for model test: ', '')
    button = st.form_submit_button("Run backtest")

if button:
    plt, backtester, portfolio_value, data = backtest.main(train_start_date, 
                                                            eval_start_date, 
                                                            eval_end_date, 
                                                            stocks,
                                                            init_capital,
                                                            mhub,
                                                            mode,
                                                            plot=True)

    for date, alloct in backtester.activity:
        idx = data.index.get_loc(date)
        plt.scatter(date, backtester.portfolio['Total'][idx]/init_capital, color='blue')
        print_alloc = f"{date.strftime(utils.DATE_FORMAT)}: "
        for k, v in alloct.items():
            if v > 0:
                print_alloc += f" Buy {v} shares of {k} "
            elif v < 0:
                print_alloc += f" Sell {v} shares of {k} "
        st.write(print_alloc)

    st.pyplot(plt, use_container_width=True)

