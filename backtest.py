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
        self.stocks = stocks

    def run_backtest(self, strategy):
        # Initialize portfolio with initial capital
        self.portfolio['Cash'] = self.initial_capital
        self.portfolio['Total'] = self.portfolio['Cash']

        for i in range(len(self.data) - 1):
            # Retrieve current date and price
            date = self.data.index[i]
            # Call strategy to determine portfolio allocation
            allocation = strategy.generate_allocation(date) # dict{stock: alloc}
            holding = 0
            cash = self.portfolio['Cash'][i]
            for stock in self.stocks:
                if len(data['Close'].columns) > 1: #track multiple stocks
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

class SimpleStrategy:
    def __init__(self, start_date, stocks):
        self.prev_date = start_date
        self.stocks = {i: 0 for i in stocks}
        
    def generate_allocation(self, date):
        delta = date.date() - self.prev_date
        if date.date() == self.prev_date:            
            return {'HES':3, 'TRGP':6, 'FANG':3, 'OKE':8, 'DHI':3, 'HCA':1, 'TSCO':1 , 'EL':1 }
        elif delta.days > 90:
            self.prev_date = date.date()
            return self.stocks
        else:
            return self.stocks


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
    start_date = "2019-12-31"  # Example start date in yyyy-mm-dd format
    end_date = "2023-05-20"  # Example end date in yyyy-mm-dd format
    stocks = ['HES', 'TRGP', 'FANG', 'OKE', 'DHI', 'HCA', 'TSCO' , 'EL', 'SPY', 'VUG', 'VOO']
    data = yf.download(stocks, start=start_date, end=end_date)
    INIT_CAPITAL = 3000

    # Create a portfolio backtester instance
    backtester = PortfolioBacktester(initial_capital=INIT_CAPITAL, data=data, stocks=stocks)
    strategy = SimpleStrategy(data.index[0].date(), stocks)

    # Run the backtest using the simple strategy
    backtester.run_backtest(strategy)

    # Retrieve the portfolio value
    portfolio_value = backtester.get_portfolio_value()
    print(portfolio_value)


    plot_cmp({"SPY":data['Close']['SPY']/data['Close']['SPY'][0], #normalize gains
            "VUG":data['Close']['VUG']/data['Close']['VUG'][0], 
            "VOO":data['Close']['VOO']/data['Close']['VOO'][0], 
            "MyPort":backtester.portfolio['Total']/INIT_CAPITAL})
