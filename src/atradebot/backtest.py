# Backtesting module for benchmarking portfolio strategies

import pandas as pd
import os
from argparse import ArgumentParser
import yfinance as yf
import matplotlib.pyplot as plt
from atradebot import strategies
from datetime import datetime
from dateutil.relativedelta import relativedelta

def get_args(raw_args=None):
    parser = ArgumentParser(description="parameters")
    parser.add_argument('--mode', type=str, default='one_stock', help='Modes: SimpleStrategy, FinForecastStrategy, FinOneStockStrategy')
    parser.add_argument('-m', '--mhub', type=str, default='achang/fin_gpt2_one_nvda', help='get from hub folder model')
    parser.add_argument('--init_capital', type=int, default=10000, help='initial capital')
    parser.add_argument('--trainstart_date', type=str, default="2019-01-31", help='train data start date')
    parser.add_argument('--evalstart_date', type=str, default="2022-01-31", help='eval start date for trading analysis')
    parser.add_argument('--evalend_date', type=str, default="2023-05-20", help='eval end data for trading analysis')
    parser.add_argument('--stocks', type=str, default="NVDA", help='stocks to analize')

    args = parser.parse_args(raw_args)
    return args


pd.options.mode.chained_assignment = None
DATE_FORMAT = "%Y-%m-%d"

class PortfolioBacktester:
    def __init__(self, initial_capital, data, stocks, start_date):
        """create backtester to evaluate portfolio strategies

        :param initial_capital: amount of money to start with
        :type initial_capital: int
        :param data: stock price data with train date and eval date. data[train_start_data:eval_end_date]
        :type data: pandas.DataFrame
        :param stocks: list of stocks to track
        :type stocks: List[str]
        :param start_date: start of evaluation period
        :type start_date: datetime.date
        """        
        self.initial_capital = initial_capital
        self.data = data
        self.portfolio = pd.DataFrame(0, index=self.data.index, columns=['Cash', 'Total'] + stocks)
        self.stocks = stocks
        self.start_date = start_date
        self.activity = [] #list of (date, allocation)


    def run_backtest(self, strategy):
        # Initialize portfolio with initial capital
        self.portfolio['Cash'] = self.initial_capital
        self.portfolio['Total'] = self.portfolio['Cash']
        idx = self.data.index.get_loc(str(self.start_date))
        for i in range(idx, len(self.data) - 1):
            # Retrieve current date and price
            date = self.data.index[i]
            # Call strategy to determine portfolio allocation
            allocation = strategy.generate_allocation(date, self.portfolio) # dict{stock: alloc} 
            holding = 0
            cash = self.portfolio['Cash'][i]
            action = False
            for stock in self.stocks:
                if len(self.data['Close'].columns) > 1: #track multiple stocks
                    price = self.data['Close'][stock][i]
                else:
                    price = self.data['Close'][i]

                # Update portfolio holdings and cash based on allocation
                if stock in allocation.keys() and cash > price*allocation[stock] and allocation[stock] != 0: #buy/sell
                    cash -= price*allocation[stock]
                    self.portfolio[stock][i+1] = self.portfolio[stock][i] + allocation[stock] 
                    action = True
                else: #hold
                    self.portfolio[stock][i+1] = self.portfolio[stock][i]
                
                holding += price*self.portfolio[stock][i+1]
                
            # Calculate total portfolio value
            if action:
                self.activity.append((date, allocation))
            self.portfolio['Cash'][i+1] = cash
            self.portfolio['Total'][i+1] = holding + cash

    def get_portfolio_value(self):
        """get portfolio value
        :return: total portfolio value
        :rtype: int
        """        
        return self.portfolio['Total']



def plot_cmp(stocks, show=False):
    # Plotting configuration
    plt.figure(figsize=(10, 6))  # Set the figure size
    for k, stock_data in stocks.items():
        plt.plot(stock_data.index, stock_data.values, label=k)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Comparison')
    plt.legend()
    if show:
        plt.show()
    return plt

def main(
        trainstart_date, 
        evalstart_date, 
        evalend_date,
        stocks,
        init_capital,
        mhub,
        mode,
        plot,
):
    """run backtest

    :param trainstart_date: train data start date
    :type trainstart_date: datetime.date
    :param evalstart_date: eval start date for trading analysis
    :type evalstart_date: datetime.date
    :param evalend_date: eval end data for trading analysis
    :type evalend_date: datetime.date
    :param stocks: lst of stocks to track
    :type stocks: list[str]
    :param init_capital: initial capital to invest
    :type init_capital: int
    :param mhub: get from hub folder model
    :type mhub: str
    :param mode: Modes: SimpleStrategy, FinForecastStrategy, FinOneStockStrategy
    :type mode: str
    :param plot: enable plot to return or not
    :type plot: bool
    :return: plot, backtester class, portfolio value, data for each stock
    :rtype: plt, backtester, portfolio_value, data
    """
    # get data:
    data = yf.download(stocks, start=trainstart_date, end=evalend_date)

    # Create a portfolio backtester instance
    backtester = PortfolioBacktester(initial_capital=init_capital, data=data, stocks=stocks, start_date=evalstart_date)

    if mode == 'SimpleStrategy':
        strategy = strategies.SimpleStrategy(evalstart_date, evalend_date, data, stocks, init_capital)
    elif mode == 'FinForecastStrategy':
        strategy = strategies.FinForecastStrategy(evalstart_date, evalend_date, data, stocks_s, init_capital, model_id=mhub)
    elif mode == 'FinOneStockStrategy':
        strategy = strategies.FinOneStockStrategy(evalstart_date, evalend_date, data, stocks_s, init_capital, model_id=mhub)
    else:
        print('Mode not recognized!')
        exit(1)

    # Run the backtest using the simple strategy
    backtester.run_backtest(strategy)

    # Retrieve the portfolio value
    portfolio_value = backtester.get_portfolio_value()
    
    if plot:
        idx = data.index.get_loc(str(evalstart_date))
        data_spy = data['Close']['SPY'][idx:]
        data_my = backtester.portfolio['Total'][idx:]
        plt = plot_cmp({"SPY":data_spy/data_spy[0], #normalize gains
                "ThisPortifolio":data_my/init_capital}, show=False)
    else:
        plt = None

    return plt, backtester, portfolio_value, data

if __name__ == "__main__":

    args = get_args()
    print('Mode for analysis:', args.mode)
    print('train start date:', args.trainstart_date)
    print('eval start date for analysis:', args.evalstart_date)
    print('eval end data for analysis:', args.evalend_date)
    stocks_s = args.stocks.split()
    stocks = stocks_s + ['SPY'] #add spy for comparison
    print('Selected stocks:', stocks_s)
    print('Extended selected stocks:', stocks)
    print('Initial capital:', args.init_capital)
    print('Model path:', args.mhub)

    trainstart_date = datetime.strptime(args.trainstart_date, format)
    evalstart_date = datetime.strptime(args.evalstart_date, format)
    evalend_date = datetime.strptime(args.evalend_date, format)
    
    plt, backtester, portfolio_value, data = main(trainstart_date, 
                                                evalstart_date, 
                                                evalend_date, 
                                                stocks, 
                                                args.init_capital, 
                                                args.mhub,
                                                args.mode, 
                                                plot=True) 

    print(portfolio_value)
    for date, alloct in backtester.activity:
        idx = data.index.get_loc(date)
        plt.scatter(date, backtester.portfolio['Total'][idx]/args.init_capital, color='blue')
        print(date, alloct)

    plt.show()