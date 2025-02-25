# Backtesting module for benchmarking portfolio strategiesallocation 

import pandas as pd
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from datetime import datetime
from atradebot.params import DATE_FORMAT, ROOT_D
from atradebot.utils.portfolio import Portfolio, test_portfolio
from atradebot.strategies import get_strategy
from atradebot.utils.utils import find_business_day, get_datetime, plot_stock
from atradebot.strategies import __MODEL_REGISTRY__
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from atradebot.strategies.strategy import StrategyConfig


pd.options.mode.chained_assignment = None

class Backtester():
    """
    A class representing a backtester for evaluating portfolio strategies.

    Attributes:
        portfolio_path (str): Path to the portfolio csv file.
        args (StrategyConfig): Strategy configuration for the backtester.
        zero (bool): Flag indicating whether to zero allocation.
        outdir (str): Output directory for the backtester results.
    """    
    def __init__(self, 
                 portfolio_path:str,
                 args:StrategyConfig,
                 zero:bool=False,
                 outdir:str='./'
                 ):
        """
        Create backtester to evaluate portfolio strategies.

        Args:
            portfolio_path (str): Path to the portfolio csv file.
            strategy (str): Strategy to use.
            past_date (datetime.date): Start of evaluation period (past).
            present_date (datetime.date): End of evaluation period.
            inference_days (datetime.date): Number of days to look back for model inference.
            outdir (str): Output directory.
        """    
        self.args = args    
        past_date = get_datetime(args.past_date)
        present_date = get_datetime(args.present_date)
        self.portfolio = Portfolio(portfolio_path, 
                                    args, 
                                    outdir=outdir)
        # for backtesting start with zero allocation to test if strategy can generate allocation and predictions.
        # Used for backtest(), findstocks(), where we start with empty portfolio
        # Only in manage() we start with a populated portfolio
        if zero:
            self.portfolio.zero_allocation()

        delta = present_date - past_date
        self.future_days = args.future_days
        self.inference_days = abs(args.inference_days) # number of days to look back for model inference
        assert delta.days >= self.inference_days, "inference_days should be less than the evaluation period"
        self.pastday = find_business_day(past_date, self.inference_days) # start of inference period
        self.strategy = get_strategy(args.strategy)(args)
        self.price_data = self.portfolio.price_data # get stock price data
        self.tickers = self.portfolio.tickers  
        self.activity = [] #list of (date, allocation)
        self.portfolio_perf = [] # total portfolio value over time
    
    def get_strategyName(self):
        """
        Get the name of the strategy class.

        Returns:
            str: The name of the strategy class.
        """        
        return self.strategy.__class__.__name__
    
    def rank_predict(self, date, future_days:list):
        """
        Predict future stock price ranks for each ticker.

        Args:
            date (str): The date for which the predictions are made.
            future_days (list): A list of future days to predict for.

        Returns:
            pandas.DataFrame: A DataFrame containing predictions for each ticker on each future day.
            contains: 'Date', 'FutureDate', 'Stock', 'Predictions', 'FuturePrice', 'Log': {"prompt", "answer", "news"}
        """        
        results = []
        for ticker in self.tickers: # for each stock
            for fday in future_days: # for eval dates
                log, pred, price = self.strategy.predict(date, fday, ticker)
                results.append({'Date':date, 'FutureDate':fday, 'Stock':ticker, 'Predictions':pred, 'FuturePrice':price, 'Log':log})
        df = pd.DataFrame(results)
        df.sort_values(by='Predictions', ascending=False, inplace=True)
        return df
        
    def eval_predict(self, plot=False):
    # measure how well model predict price change for stocks
        """
        Evaluate and predict the stock prices using a given strategy.

        Args:
            plot (bool, optional): Whether to plot and save the predictions. Default is False.

        Returns:
            list: A list of dictionaries containing evaluation metrics and predictions for each stock.

        Raises:
            None
        """        
        idx = self.price_data.index.get_loc(str(self.pastday))
        results = []
        # num_evals = int(len(self.price_data)*0.2)
        num_evals = len(self.price_data)
        for ticker in self.tickers: # for each stock
            tgts, preds = [], []
            for i in range(idx, num_evals - 1): # for eval dates
                date = self.price_data.index[i]
                tgt = self.price_data.iloc[i+1]['Close'][ticker]
                _, pred, price = self.strategy.predict(date, self.future_days, ticker)
                tgts.append(tgt)
                preds.append(price)
            MSE = mean_squared_error(preds, tgts)
            RMSE = MSE**0.5
            MAE = mean_absolute_error(preds, tgts)
            R2 = r2_score(preds, tgts)
            if plot:
                plt.plot(tgts, label=f'Target_{ticker}')
                plt.plot(preds, label=f'Predicted_{ticker}')
                plt.legend()
                plt.savefig(f'{ROOT_D}/sd/predict_{ticker}_{str(self.strategy.__class__.__name__)}.png')
                plt.close()
            results.append({'Stock':ticker, 'MSE': MSE, 'RMSE': RMSE, 'MAE': MAE, 'R2': R2, 'targets':tgts, 'predictions':preds})
    
        return results

    def run_backtest(self):
    # measure how well the strategy performs to trade and allocate stocks 
        # Initialize portfolio with initial capital
        """
        Run backtest on a financial strategy using historical price data.

        Args:
            self: The object instance.
    
        Returns:
            None

        Note:
            This function updates the portfolio based on the generated allocation from the strategy for each date in the historical price data.

        Raises:
            None
        """        
        idx = self.price_data.index.get_loc(str(self.pastday))
        for i in range(idx, len(self.price_data) - 1):
            # Retrieve current date and price
            date = self.price_data.index[i]
            # Call strategy to determine portfolio allocation
            allocation = self.strategy.generate_allocation(date, self.portfolio) # dict{stock: alloc} 
            self.portfolio.update_portfolio(date)
            act = self.portfolio.execute(allocation)
            # Calculate total portfolio value
            if act:
                self.activity.append((date, allocation))
            self.portfolio_perf.append({'Date':date, 'Price': self.portfolio.get_portfolio_value()})

        assert len(self.portfolio_perf) > 0, "run_backtest: no portfolio performance data available"

        self.portfolio_perf = pd.DataFrame(self.portfolio_perf).set_index('Date')

def backtest(
        portfolio_path:str,
        args:StrategyConfig,
        eval_mode:str='predict',
        plot:bool=False,
):
    """
    run backtest

    Args:
        plot (bool): Enable plot to return or not.

    Returns:
        tuple: plot, backtester class, portfolio value, data for each stock.

    Raises:
        None
    """
    # Create a portfolio backtester instance
    init_capital = args.cash
    if not portfolio_path: # TODO: no search new stocks backtesting yet
        stocks=['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL', 'AMZN']
        portfolio_path = test_portfolio(stocks)

    backtester = Backtester(portfolio_path, args=args, zero=True)
    if eval_mode == 'predict':
        backtester.eval_predict(plot=True)
        return
    
    # Run the backtest using the simple strategy
    backtester.run_backtest()

    # Retrieve the portfolio value
    portfolio_value = backtester.portfolio.get_portfolio_value()
    
    if plot:
        idx = backtester.portfolio.price_data.index.get_loc(str(backtester.pastday))
        data_spy = backtester.portfolio.benchmark_price['Close'][idx:]
        data_my = backtester.portfolio_perf[idx:]
        cost = backtester.portfolio.get_cost()
        qnt = int(cost/data_spy.values[0][0])
        # dict of pandas series: Date: Price
        plt = plot_stock({
                "SPY":data_spy*qnt, #normalize gains
                "ThisPortfolio":data_my}, show=False)
        plt.savefig(f'{ROOT_D}/sd/backtest_{backtester.get_strategyName()}.png')
        print('Activity: ', backtester.activity)
    else:
        plt = None

    print(portfolio_value)
    print(backtester.portfolio)

def get_args(raw_args=None):
    """
    Get the command line arguments for the program.

    Args:
        raw_args (list): A list of raw arguments to parse. Default is None.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.

    Raises:
        None.

    Description:
        This function creates an ArgumentParser object to define and parse command line arguments
        for the program. It sets various optional arguments for the strategy, number of days for inference,
        number of days in the future to predict, start and end dates for trading analysis, evaluation mode,
        and path to the portfolio. It then parses the raw arguments provided and returns an object
        containing the parsed arguments.

        Example Usage:
        raw_args = ['--strategy', 'RandomStrategy', '--inference_days', '3']
        args = get_args(raw_args)
    """    
    parser = ArgumentParser(description="parameters")
    parser.add_argument('--strategy','-s', type=str, default='ManualStrategy', help=f'Modes: {__MODEL_REGISTRY__.keys()}')
    parser.add_argument('--inference_days','-i', type=int, default=1, help='number of days for inference')
    parser.add_argument('--future_days','-f', type=int, default=1, help='number of days in the future to predict')
    parser.add_argument('--past_date', type=str, default="2024-04-01", help='eval start date for trading analysis (past)')
    parser.add_argument('--present_date', type=str, default="2025-01-30", help='eval end data for trading analysis')
    parser.add_argument('--newsapi','-n', type=str, default="dataset", help='newsapi: finhub, newsapi, google')
    parser.add_argument('--eval_mode', type=str, default="allocation", help='eval modes: predict and allocation')
    parser.add_argument('--portfolio_path', type=str, default="", help='path to portfolio')
    args = parser.parse_args(raw_args)
    return args

if __name__ == "__main__":

    args = get_args()
    print(args)
    past_date = datetime.strptime(args.past_date, DATE_FORMAT).date()
    present_date = datetime.strptime(args.present_date, DATE_FORMAT).date()
    config = StrategyConfig(
        past_date=past_date,
        present_date=present_date,
        inference_days=args.inference_days,
        future_days=args.future_days,
        strategy=args.strategy,
        newsapi=args.newsapi,
    )

    backtest(
        portfolio_path=args.portfolio_path,
        args=config,
        eval_mode=args.eval_mode,
        plot=True)
    
