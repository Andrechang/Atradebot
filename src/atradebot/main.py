import pandas as pd
from argparse import ArgumentParser
import os
from datetime import date, datetime, timedelta
import shutil
from atradebot.params import DATE_FORMAT, ROOT_D, logging
from atradebot.strategies import __MODEL_REGISTRY__
from atradebot.utils.backtest import Backtester  
from atradebot.utils.utils import find_business_day, gen_report
from atradebot.utils.search import findstocks
from atradebot.strategies.strategy import StrategyConfig
from atradebot.utils.utils import ALLTICKS
from atradebot.utils.alpaca_trade import alpaca_execute
from atradebot.utils.portfolio import test_portfolio
  
# check current portfolio stocks that could go up/down
def manage(args:StrategyConfig, portfolio_path):
    # get current portfolio
    """
    Manage a portfolio using a given strategy configuration.

    Args:
        args (StrategyConfig): An object containing configuration settings for the strategy.
        portfolio_path (str): The file path to the portfolio data.

    Returns:
        None

    Calls:
        Backtester: An instance of the Backtester class for managing the portfolio.
        gen_report: A function to generate a report based on the backtester results.

    Raises:
        N/A
    """    
    backtester = Backtester(portfolio_path, args=args, zero=False)   
    future_days = [30, 120, 360]
    pred_result = backtester.rank_predict(args.present_date, future_days)
    gen_report(backtester, pred_result, 'manage')
    if args.run_trade:
        action = backtester.strategy.generate_allocation()
        alpaca_execute(action)
    return backtester

def main(args:StrategyConfig):
    """
    Execute the main strategy based on the given configuration.

    Args:
        args (StrategyConfig): An object containing the necessary configuration parameters.
    
    Returns:
        None
    """    
    args.present_date = date.today()
    args.past_date = find_business_day(args.present_date, -args.inference_days)

    # check current portfolio stocks
    portfolio_path = f'{ROOT_D}/sd/test_portfolio.csv' 
    backt = manage(args, portfolio_path)

    # search for new stocks to try
    findstocks(args, verbose=True, skip_tickers=backt.portfolio.tickers) # stocks to buy (good stocks)
    # skip tickers that are already in the portfolio

    findstocks(args, verbose=True, sell=True) # stocks to sell (bad stocks)

def get_args(raw_args=None):
    """
    Get command line arguments for the script.

    Args:
        raw_args (list): A list of raw arguments from the command line. Default is None.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.

    Raises:
        None.
    """    
    parser = ArgumentParser(description="parameters")
    parser.add_argument('--strategy','-s', type=str, default='GPTStrategy', help=f'Modes: {__MODEL_REGISTRY__.keys()}')
    parser.add_argument('--inference_days','-i', type=int, default=30, help='number of days for inference')
    parser.add_argument('--future_days','-f', type=int, default=30, help='number of days in the future to predict')
    parser.add_argument('--newsapi','-n', type=str, default="finhub", help='newsapi: finhub, newsapi, google')
    args = parser.parse_args(raw_args)
    return args


if __name__ == "__main__":
    args = get_args()
    print(args)
    assert args.strategy != 'ManualStrategy', f"Invalid strategy: {args.strategy}"
    config = StrategyConfig(
        inference_days=args.inference_days,
        future_days=args.future_days,
        strategy=args.strategy,
        newsapi=args.newsapi
    )
    main(config)
