import pandas as pd
import os
from datetime import date, datetime
import yfinance as yf
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from pypfopt import risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.efficient_frontier import EfficientFrontier

pd.options.mode.chained_assignment = None

class PortfolioBacktester:
    def __init__(self, initial_capital, data, stocks, start_date):
        self.initial_capital = initial_capital
        self.data = data
        self.portfolio = pd.DataFrame(0, index=self.data.index, columns=['Cash', 'Total'] + stocks)
        self.stocks = stocks
        self.start_date = start_date

    def run_backtest(self, strategy):
        # Initialize portfolio with initial capital
        self.portfolio['Cash'] = self.initial_capital
        self.portfolio['Total'] = self.portfolio['Cash']
        idx = self.data.index.get_loc(self.start_date)
        for i in range(idx, len(self.data) - 1):
            # Retrieve current date and price
            date = self.data.index[i]
            # Call strategy to determine portfolio allocation
            allocation = strategy.generate_allocation(date) # dict{stock: alloc}
            holding = 0
            cash = self.portfolio['Cash'][i]
            for stock in self.stocks:
                if len(self.data['Close'].columns) > 1: #track multiple stocks
                    price = self.data['Close'][stock][i]
                else:
                    price = self.data['Close'][i]

                # Update portfolio holdings and cash based on allocation
                if stock in allocation.keys() and cash > price*allocation[stock]: #buy/sell
                    cash -= price*allocation[stock]
                    self.portfolio[stock][i+1] = self.portfolio[stock][i] + allocation[stock] 
                else: #hold
                    self.portfolio[stock][i+1] = self.portfolio[stock][i]
                
                holding += price*self.portfolio[stock][i+1]
                
            # Calculate total portfolio value
            self.portfolio['Cash'][i+1] = cash
            self.portfolio['Total'][i+1] = holding + cash

    def get_portfolio_value(self):
        return self.portfolio['Total']



# Define a simple strategy that allocates 50% of the portfolio on the first day
def max_sharpe_allocation(data, amount_invest):
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    ef = EfficientFrontier(mu, S)
    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    latest_prices = get_latest_prices(data)
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=amount_invest)
    allocation, leftover = da.greedy_portfolio()
    return allocation, leftover

class SimpleStrategy:
    def __init__(self, start_date, end_date, data, stocks, cash=10000):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        self.prev_date = start_date
        self.stocks = {i: 0 for i in stocks}
        self.cash = [cash*0.5, cash*0.3, cash*0.1, cash*0.1] #amount to invest in each rebalance
        self.start_date = start_date
        self.end_date = end_date
        delta = end_date - start_date
        self.days_interval = delta.days/len(self.cash) #rebalance 4 times
        self.data = data['Adj Close']
        self.cash_idx = 0

    def generate_allocation(self, date):
        delta = date.date() - self.prev_date
        idx = self.data.index.get_loc(str(date.date()))
        if date.date() == self.prev_date: #first day
            allocation, leftover = max_sharpe_allocation(self.data[0:idx], amount_invest=self.cash[self.cash_idx])
            self.cash_idx += 1                        
            return allocation
        elif delta.days > self.days_interval: #rebalance 
            allocation, leftover = max_sharpe_allocation(self.data[0:idx], amount_invest=self.cash[self.cash_idx])
            self.prev_date = date.date()
            self.cash_idx += 1
            return allocation
        else:
            return self.stocks


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

if __name__ == "__main__":
    
    # Example usage
    past_date = "2018-01-31" 
    start_date = "2019-01-31"  
    end_date = "2020-05-20" 
    stocks = ['AAPL','ABBV','AMZN','MSFT','NVDA','TSLA', 'SPY', 'VUG', 'VOO']
    data = yf.download(stocks, start=past_date, end=end_date)
    INIT_CAPITAL = 10000

    # Create a portfolio backtester instance
    backtester = PortfolioBacktester(initial_capital=INIT_CAPITAL, data=data, stocks=stocks, start_date=start_date)
    strategy = SimpleStrategy(start_date, end_date, data, stocks, INIT_CAPITAL)

    # Run the backtest using the simple strategy
    backtester.run_backtest(strategy)

    # Retrieve the portfolio value
    portfolio_value = backtester.get_portfolio_value()
    print(portfolio_value)

    idx = data.index.get_loc(start_date)
    data_spy = data['Close']['SPY'][idx:]
    data_my = backtester.portfolio['Total'][idx:]
    plot_cmp({"SPY":data_spy/data_spy[0], #normalize gains
            "MyPort":data_my/INIT_CAPITAL}, show=True)
