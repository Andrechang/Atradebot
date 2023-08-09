import streamlit as st
from atradebot import backtest, utils, fin_train
import yfinance as yf
import pandas as pd

"""
# Test model
"""


past_date = "2019-01-31" 
start_date = "2022-01-31"  
end_date = "2023-05-20" 
stocks = ['SPY']

INIT_CAPITAL = 10000
stocks_s = st.text_input("select stock", value="TSLA")
date_s = st.text_input("select time", value="2021-07-22")

stocks.append(stocks_s)
data = yf.download(stocks, start=past_date, end=end_date)

strategy = fin_train.FinForecastStrategy(start_date, end_date, data, [stocks_s], INIT_CAPITAL)

date_p = pd.Timestamp(date_s)
print(date_p)
pred = strategy.model_run(date_p)
pred = pred[stocks_s]
date_ss = date_p.strftime("%b %d, %Y")
target = utils.get_forecast(stocks_s, date_ss)

st.write(f"Predicted forecast for {date_s}: \n 1 mon: {pred[0]} 5 mon: {pred[1]} 1 year: {pred[2]}")
st.write(f"Actual forecast: 1 mon: {target[0]} 5 mon: {target[1]} 1 year: {target[2]}")

plt = backtest.plot_cmp({"TSLA":data['Close']['TSLA']/data['Close']['TSLA'][0]})
idx = data.index.get_loc(date_s)
plt.scatter(date_p, data['Close']['TSLA'][idx]/data['Close']['TSLA'][0], color='red')
st.pyplot(plt, use_container_width=True)



