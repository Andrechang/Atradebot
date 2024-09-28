# technical analysis functions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from datetime import datetime, date
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override() # https://github.com/pydata/pandas-datareader/issues/952

import requests_cache
session = requests_cache.CachedSession(cache_name='cache', backend='sqlite')

from utils import find_business_day

def StochRSI(series, period=14, smoothK=3, smoothD=3):
    # Calculate RSI 
    delta = series.diff().dropna()
    ups = delta * 0
    downs = ups.copy()
    ups[delta > 0] = delta[delta > 0]
    downs[delta < 0] = -delta[delta < 0]
    ups[ups.index[period-1]] = np.mean( ups[:period] ) #first value is sum of avg gains
    ups = ups.drop(ups.index[:(period-1)])
    downs[downs.index[period-1]] = np.mean( downs[:period] ) #first value is sum of avg losses
    downs = downs.drop(downs.index[:(period-1)])
    rs = ups.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() / \
         downs.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() 
    rsi = 100 - 100 / (1 + rs)

    # Calculate StochRSI 
    stochrsi  = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    stochrsi_K = stochrsi.rolling(smoothK).mean()
    stochrsi_D = stochrsi_K.rolling(smoothD).mean()

    return stochrsi, stochrsi_K, stochrsi_D

def isSupport(df,i):
    support = df['Low'][i] < df['Low'][i-1]  and df['Low'][i] < df['Low'][i+1] \
    and df['Low'][i+1] < df['Low'][i+2] and df['Low'][i-1] < df['Low'][i-2]
    return support

def isResistance(df,i):
    resistance = df['High'][i] > df['High'][i-1]  and df['High'][i] > df['High'][i+1] \
    and df['High'][i+1] > df['High'][i+2] and df['High'][i-1] > df['High'][i-2] 

    return resistance

def test_support_resistance(data):
    levels = []
    s =  np.mean(data['High'] - data['Low'])
    for i in range(2, data.shape[0]-2):
        if isSupport(data,i):
            l = data['Low'][i]
            if np.sum([abs(l-x) < s  for x in levels]) == 0:
                levels.append((i,l))
        elif isResistance(data,i):
            l = data['High'][i]
            if np.sum([abs(l-x) < s  for x in levels]) == 0:
                levels.append((i,l))

    plt.plot(data.index, data['Close'].values)
    for level in levels:
        plt.scatter(data.index[level[0]], level[1], color='red')

def test_stochRSI(data):
    s,_,_= StochRSI(data['Close'])
    condition = s > 0.8
    filtered_index = s.loc[condition].index
    condition = s < 0.2
    filtered_index_2 = s.loc[condition].index
    plt.plot(data.index, data['Close'].values)
    for level in filtered_index:
        plt.scatter(level, data['Close'][level], color='red', )
    for level in filtered_index_2:
        plt.scatter(level, data['Close'][level], color='blue', )

class Portfolio():
    def __init__(self, portifolio_path, backdays=-360) -> None:
        self.today = find_business_day(date.today(), -1)
        self.pastday = find_business_day(self.today, backdays) # get date 1 year ago

        stocks = pd.read_csv(portifolio_path)# read in list of stocks
        stocks_lst = stocks[stocks['Quantity'].notnull()]
        stocks_lst['Ticker'] = stocks_lst['Ticker'].fillna('#CD')# fill NaN with #<description> ticker
        self.portifolio = stocks_lst
        self.tickers = stocks_lst['Ticker'].tolist()
        self.cost = stocks_lst['Cost'].tolist()
        self.weights = self.cost/np.sum(self.cost)
 
        price_data = pdr.get_data_yahoo(self.tickers, start=self.pastday, end=self.today, session=session) 
        self.price_data = price_data.fillna(1) # CD and MM value is 1
        self.benchmark_price = pdr.get_data_yahoo('SPY',start=self.pastday, end=self.today, session=session)

    def calc_beta(self):
        """calculate beta and alpha of a portfolio against the S&P 500
        :return: beta, alpha
        :rtype: tuple
        """    
        price_data_adj = self.price_data['Adj Close']
        ret_data = price_data_adj.pct_change()[1:]
        port_ret = (ret_data * self.weights).sum(axis = 1)
        
        benchmark_ret = self.benchmark_price["Adj Close"].pct_change()[1:]
        (beta, alpha) = stats.linregress(benchmark_ret.values, port_ret.values)[0:2]
        # print("The portfolio beta is", round(beta, 4))
        # print("The portfolio alpha is", round(alpha,5))
        return beta, alpha

if __name__ == "__main__":
    # data = yf.download('AAPL', start="2020-03-15", end="2022-07-15")
    # test_stochRSI(data)

    portifolio = Portfolio('')
    beta, alpha = portifolio.calc_beta()
    print(beta, alpha)
