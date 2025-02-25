# technical analysis functions
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from argparse import ArgumentParser
from datetime import datetime, date
import random
from atradebot.params import DATE_FORMAT, ROOT_D
from atradebot.utils.utils import find_business_day, get_datetime
from atradebot.data.yf_data import get_yf_data_some, get_price


class Portfolio():
    """
    A class representing a portfolio.

    Attributes:
        portfolio_path (str): The path to the portfolio csv file.
        past_date (datetime.date): The start date for price data collection (past).
        present_date (datetime.date): The end date for price data collection.
        outdir (str): The output directory path.
    """    
    def __init__(self, 
                 portfolio_path:str, 
                 args,
                 outdir='./') -> None:
        
        '''
        Args:
            portfolio_path (str): path to the portfolio csv file
            past_date (datetime.date): start of date for price data collect (past)
            present_date (datetime.date): end of date for price data collect
        '''
        past_date = get_datetime(args.past_date)
        present_date = get_datetime(args.present_date)

        # load portfolio
        self.outdir = outdir
        self.infile = portfolio_path
        # pd.DataFrame columns: ['Description', 'Ticker', 'Quantity', 'Price', 'Cost', 'Unit Cost', 'Value', 'Gain', 'Gain %']
        
        stocks = pd.read_csv(self.infile)
        stocks_lst = stocks[stocks['Quantity'].notnull()]
        stocks_lst = stocks_lst.dropna()
        stocks_lst['Ticker'] = stocks_lst['Ticker'].str.strip()
        self.tickers = stocks_lst['Ticker'].tolist()
        self.portfolio = stocks_lst # current portfolio
        cols2float = ['Price', 'Cost', 'Unit Cost', 'Value', 'Gain', 'Gain %']
        self.portfolio[cols2float] = self.portfolio[cols2float].astype(float)

        # Get price DATA
        self.today = find_business_day(present_date)
        self.pastday = find_business_day(past_date)
        self.evalday = self.today # update_portfolio will change the portfolio evaluation date
        self.get_data(self.pastday, self.today) 
        # sets: self.today, self.pastday, self.price_data, self.benchmark_price

        self.cash_available = args.cash # cash available for trading

    def __str__(self):
        """
        Return a string representation of the object.

        This method calculates and prints the portfolio beta, alpha and value.
        It also returns a string representation of the portfolio.

        Returns:
            str: A string representation of the portfolio.
        """        
        portfolio_value = self.get_portfolio_value()
        beta, alpha = self.calc_beta()
        print("The portfolio beta is ", round(beta, 4)," alpha is ", round(alpha,5))
        print(f"The portfolio value is {portfolio_value}")
        return self.portfolio.to_string()
    
    def get_cost(self):
        """
        Get portfolio cost.

        Returns:
            int: Total portfolio cost.
        """        
        return self.portfolio['Cost'].sum()
    
    def get_portfolio_value(self):
        """
        Get portfolio value.

        Returns:
            int: Total portfolio value.
        """        
        return self.portfolio['Value'].sum()

    def get_data(self, pastday:datetime.date, presentday:datetime.date):
        """
        Get price data for a list of tickers within a specified date range.

        Args:
            pastday (datetime.date): The start date for the data retrieval.
            presentday (datetime.date): The end date for the data retrieval.

        Returns:
            None

        Side Effects:
            - Updates the 'price_data' attribute of the object with fetched data for the tickers.
            - Drops any rows with missing values from the 'price_data'.
            - Updates the 'benchmark_price' attribute with data fetched for the S&P 500 index (ticker 'SPY').
            - Calls the 'update_portfolio' method to update the portfolio based on the retrieved data.
        """        
        self.price_data = get_yf_data_some(self.tickers, past_date=pastday, present_date=presentday) 
        self.price_data = self.price_data.dropna()
        self.benchmark_price = get_yf_data_some('SPY', past_date=pastday, present_date=presentday)
        self.update_portfolio()

    def zero_allocation(self):
        """
        Zero out the allocation values in the portfolio.

        This function sets all allocation values in the portfolio to zero.

        Args:
            self: The object instance representing the portfolio.

        Returns:
            None
        """        
        self.portfolio['Quantity'] = 0
        self.portfolio['Cost'] = 0
        self.portfolio['Unit Cost'] = 0
        self.portfolio['Value'] = 0
        self.portfolio['Gain'] = 0
        self.portfolio['Gain %'] = 0

# Technical Analysis --------------------------------------------
    def calc_beta(self):
        """
        Calculate beta and alpha of a portfolio against the S&P 500.

        Returns:
            tuple: beta, alpha
        """    
        cost = self.portfolio['Cost'].tolist()
        weights = cost/np.sum(cost)

        price_data_adj = self.price_data['Close']
        ret_data = price_data_adj.pct_change()[1:]
        port_ret = (ret_data * weights).sum(axis = 1)
        
        benchmark_ret = self.benchmark_price["Close"].pct_change()[1:]
        b_v = np.squeeze(benchmark_ret)
        (beta, alpha) = stats.linregress(b_v, port_ret.values)[0:2]

        return beta, alpha
    
# Flags ---------------------------------------------------------
    def stop_loss(self, loss=0.4):
        # check if the stock goes down loss_percent from the cost basis 
        """
        Calculate the stop loss levels for the portfolio.

        Args:
            loss (float): The percentage loss at which the stop loss should trigger. Default is 0.4.

        Returns:
            dict: A dictionary containing information on the assets in the portfolio that have reached the stop loss level.

        Raises:
            None
        """        
        stop_loss = self.portfolio['Unit Cost'] * (1 - loss)
        flag = self.portfolio['Price'] < stop_loss
        return self.portfolio[flag]
    
    def stop_down(self, days=5):
        # check if the stock is going down for days consecutively 
        """
        Calculate the percentage change in exponential moving average (EMA) with respect to the maximum EMA value over a specified number of days.

        Args:
            self: The object instance.
            days (int): Number of days to consider for the EMA calculation. Default is 5.

        Returns:
            pandas.Series: A pandas Series containing the percentage change in EMA values sorted in ascending order.
        """        
        data_ema = self.price_data['Close'].ewm(span=days, adjust=False).mean() # EMA
        data_ema_max = data_ema.max()
        # get ema percentage change for the last days
        ema_pct_change = (data_ema.iloc[-1]-data_ema_max)/data_ema.iloc[-1]
        ema_pct_change.sort_values(ascending=True, inplace=True)
        return ema_pct_change

# Portfolio management -----------------------------------------
    def buy_stock(self, stock:str, quantity:int):
        # add stock to the portfolio
        """
        Buy a stock and update the portfolio with the transaction details.

        Args:
            stock (str): The stock symbol to buy.
            quantity (int): The quantity of the stock to buy.

        Returns:
            None

        Raises:
            None
        """        
        day = self.evalday.strftime(DATE_FORMAT)
        if stock in self.tickers: # add to existing stock
            idx = self.portfolio[self.portfolio['Ticker'] == stock].index
            price = self.portfolio.loc[idx, 'Price']
            self.portfolio.loc[idx, 'Quantity'] += quantity
            self.portfolio.loc[idx, 'Cost'] += price * quantity
            self.portfolio.loc[idx, 'Value'] += price*quantity
            self.portfolio.loc[idx, 'Gain'] = self.portfolio['Value'][idx] - self.portfolio['Cost'][idx]
            self.portfolio.loc[idx, 'Gain %'] = self.portfolio['Gain'][idx]/self.portfolio['Cost'][idx]
            self.portfolio.loc[idx, 'Unit Cost'] = self.portfolio['Cost'][idx]/self.portfolio['Quantity'][idx]
        else: # new stock append to portfolio
            new_prices = get_yf_data_some(stock, past_date=self.pastday, present_date=self.today)
            price = new_prices['Close'].loc[day]
            self.price_data =  pd.concat([self.price_data, new_prices],axis=1)
            self.tickers.append(stock)
            cost = price * quantity
            new_stock = pd.DataFrame({'Description': stock, 'Ticker': stock, 'Quantity': quantity, 
                                    'Price': price, 'Cost': cost, 'Unit Cost': cost/quantity, 
                                    'Value': price*quantity, 'Gain': price*quantity - cost, 'Gain %': (price*quantity - cost)/cost})
            self.portfolio = pd.concat([self.portfolio, new_stock])
    
    def sell_stock(self, stock:str, quantity:int):
        # sell stock from the portfolio
        """
        Sell a specific stock from the portfolio.

        Args:
            stock (str): The ticker symbol of the stock to be sold.
            quantity (int): The quantity of the stock to be sold.

        Returns:
            float: The total revenue generated from selling the stock.

        Raises:
            None
        """        
        idx = self.portfolio[self.portfolio['Ticker'] == stock].index
        if idx.empty:
            # print(f"Stock {stock} not found in portfolio")
            return 0
        revenue = quantity * self.portfolio.loc[idx, 'Price']
        if quantity >= self.portfolio['Quantity'][idx[-1]]: # sell all
            self.portfolio = self.portfolio.drop(idx)
            self.tickers.remove(stock)
            self.price_data.drop(stock, axis=1, inplace=True)
        else:
            self.portfolio.loc[idx,'Quantity'] -= quantity
            self.portfolio.loc[idx,'Cost'] -= self.portfolio['Unit Cost'][idx]*quantity
            self.portfolio.loc[idx,'Value'] -= self.portfolio['Price'][idx]*quantity
            self.portfolio.loc[idx,'Gain'] = self.portfolio['Value'][idx] - self.portfolio['Cost'][idx]
            self.portfolio.loc[idx,'Gain %'] = self.portfolio['Gain'][idx]/self.portfolio['Cost'][idx]
        return revenue
    
    def update_portfolio(self, day:datetime.date=None):
        # update portfolio
        """
        Update the portfolio with the latest price data.

        Args:
            day (datetime.date, optional): The date of evaluation. If None, the latest available date will be used.

        Returns:
            None

        Updates:
            - 'Price', 'Value', 'Gain', and 'Gain %' columns in the portfolio DataFrame.

        Raises:
            KeyError: If the close price data for the specified day is not available.
        """        
        if day:
            self.evalday = day
            day = day.strftime(DATE_FORMAT)
            d = self.price_data['Close'].loc[day].to_dict()
        else:
            d = self.price_data['Close'].iloc[-1].to_dict()
        self.portfolio['Price'] = self.portfolio["Ticker"].map(d).fillna(self.portfolio["Price"])
        self.portfolio['Value'] = self.portfolio['Price'] * self.portfolio['Quantity']
        self.portfolio['Gain'] = self.portfolio['Price'] * self.portfolio['Quantity'] - self.portfolio['Cost']
        self.portfolio['Gain %'] = (self.portfolio['Price'] * self.portfolio['Quantity'] - self.portfolio['Cost']) / self.portfolio['Cost']

    def save_back(self):
        """
        Save the portfolio data to a CSV file.

        Creates the output directory if it does not exist and saves the portfolio data to a CSV file.

        Raises:
            OSError: If the output directory cannot be created or the CSV file cannot be saved.
        """        
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
        outfile = os.path.join(self.output_dir,'portfolio', datetime.now().strftime("%b%d_%H-%M-%S"), '.csv')
        self.portfolio.to_csv(outfile, index=False)

    def plot_portfolio(self):
        # plot portfolio
        """
        Plot the portfolio values for each ticker.

        This method plots the prices for each ticker in the portfolio in a bar chart.

        Args:
            self: The Portfolio object instance.

        Returns:
            None

        Note:
            This method requires the 'Ticker' and 'Price' columns in the portfolio DataFrame.

        Example:
            portfolio.plot_portfolio()
        """        
        self.portfolio.plot(x='Ticker', y='Price', kind='bar')
        plt.show()

    def execute(self, action:dict):
        # execute
        act = False
        for stock, qnt in action.items():
            # buy
            if qnt > 0:
                self.buy_stock(stock, qnt)
                act = True
            # sell
            elif qnt < 0:
                self.sell_stock(stock, qnt)
                act = True
        return act


def dict2portfolio(portfolio:dict):
    """
    Convert a dictionary to a portfolio and save it to a CSV file.

    Args:
        portfolio (dict): A dictionary containing stock tickers and their (respective quantities, cost basis).
        outfile (str): The output file path for the CSV file.

    Returns:
        str: The file path of the created portfolio.

    Raises:
        None.
    """    
    portfolio_path = f'{ROOT_D}/sd/tmp_.csv'
    with open(portfolio_path, 'w') as f:
        f.write('Description,Ticker,Quantity,Price,Cost,Unit Cost,Value,Gain,Gain %\n')
        for stk in portfolio.keys():
            price = get_price(stk)
            qnt = portfolio[stk][0]
            cost = portfolio[stk][1]
            f.write(f'{stk},{stk},{qnt},{qnt*price},{cost},{cost/qnt},{qnt*price},{qnt*price-cost},{(qnt*price-cost)/cost}\n')
    return portfolio_path


def test_portfolio(stklist:list, rnd=False):
    """
    Create a test portfolio based on a list of stocks.

    Args:
        stklist (list): A list of stock tickers.
        rnd (bool): If True, generate random quantities for the stocks. Default is False.

    Returns:
        str: The file path of the created portfolio.

    Raises:
        None.
    """    
    portfolio_path = f'{ROOT_D}/sd/tmp.csv'
    with open(portfolio_path, 'w') as f:
        f.write('Description,Ticker,Quantity,Price,Cost,Unit Cost,Value,Gain,Gain %\n')
        for stk in stklist:
            price = get_price(stk)
            if rnd:
                qnt = random.randint(1, 5)
            else:
                qnt = 0
            f.write(f'{stk},{stk},{qnt},{qnt*price},{qnt*price},{price},0,0,0\n')
    return portfolio_path

if __name__ == "__main__":
    portfolio_path = f'{ROOT_D}/sd/test_portfolio.csv'
    portfolio = Portfolio(portfolio_path, past_date='2023-04-05', present_date='2023-08-13')
    beta, alpha = portfolio.calc_beta()
    print("The portfolio beta is", round(beta, 4))
    print("The portfolio alpha is", round(alpha,5))
    bdays = 3
    sloss = 0.4
    print(f"The stocks going down past {bdays} days: ", portfolio.stop_down(bdays))
    print(f"The stocks went down {sloss} from cost basis are: ", portfolio.stop_loss(loss=sloss))
    
    portfolio.update_portfolio(day=get_datetime('2023-08-07'))
    print(portfolio)
    portfolio.buy_stock('AAPL', 10)
    print(portfolio)
    portfolio.sell_stock('AAPL', 5)
    print(portfolio)
    


    