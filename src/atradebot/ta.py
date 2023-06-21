# technical analysis functions
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

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

if __name__ == "__main__":
    data = yf.download('AAPL', start="2020-03-15", end="2022-07-15")
    test_stochRSI(data)