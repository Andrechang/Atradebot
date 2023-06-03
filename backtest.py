import pandas as pd
import os
from datetime import date, datetime
import yfinance as yf
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt


class PortfolioBacktester:
    def __init__(self, initial_capital, data, stocks):
        self.initial_capital = initial_capital
        self.data = data
        self.portfolio = pd.DataFrame(0, index=data.index, columns=['Cash', 'Total'] + stocks)
        
    def run_backtest(self, strategy):
        # Initialize portfolio with initial capital
        self.portfolio['Cash'] = self.initial_capital
        self.portfolio['Total'] = self.portfolio['Cash']

        for i in range(len(self.data) - 1):
            # Retrieve current date and price
            date = self.data.index[i]
            # Call strategy to determine portfolio allocation
            allocation = strategy.generate_allocation(date, self.portfolio) # {stock: alloc}
            
            holding = 0
            cash = self.portfolio['Cash'][i]
            for stock, stock_alloc in allocation.items():
                if len(self.data['Close'].columns) > 1: #track multiple stocks
                    price = self.data['Close'][stock][i]
                else:
                    price = self.data['Close'][i]
                # Update portfolio holdings and cash based on allocation
                if cash > price*stock_alloc:
                    cash -= price*stock_alloc
                    self.portfolio[stock][i+1] = self.portfolio[stock][i] + stock_alloc 
                else:
                    self.portfolio[stock][i+1] = self.portfolio[stock][i]
                
                holding += price*self.portfolio[stock][i+1]
                
            # Calculate total portfolio value
            self.portfolio['Cash'][i+1] = cash
            self.portfolio['Total'][i+1] = holding + cash

    def get_portfolio_value(self):
        return self.portfolio['Total']



# Define a simple strategy that allocates 50% of the portfolio on the first day

class SimpleStrategy:
    def __init__(self, start_date):
        self.prev_date = start_date
    def generate_allocation(self, date, portfolio):
        delta = date.date() - self.prev_date
        if date.date() == self.prev_date:            
            return {"AAPL":0, "SPY":10}
        elif delta.days > 90:
            self.prev_date = date.date()
            return {"AAPL":0, "SPY":5}
        else:
            return {"AAPL":0, "SPY":0}


def plot_cmp(stocks):
    # Plotting configuration
    plt.figure(figsize=(10, 6))  # Set the figure size
    for k, stock_data in stocks.items():
        plt.plot(stock_data.index, stock_data.values, label=k)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Comparison')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Example usage
    start_date = "2020-01-01"  # Example start date in yyyy-mm-dd format
    end_date = "2023-04-20"  # Example end date in yyyy-mm-dd format
    stocks = ['AAPL', 'SPY']
    data = yf.download(stocks, start=start_date, end=end_date)


    # Create a portfolio backtester instance
    backtester = PortfolioBacktester(initial_capital=50000, data=data, stocks=stocks)

    # Run the backtest using the simple strategy
    backtester.run_backtest(SimpleStrategy(data.index[0].date()))

    # Retrieve the portfolio value
    portfolio_value = backtester.get_portfolio_value()
    print(portfolio_value)

    plot_cmp({"SPY":data['Close']['SPY']/data['Close']['SPY'][0], "MyPort":backtester.portfolio['Total']/50000})