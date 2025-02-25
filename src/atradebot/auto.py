
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

def auto_trade(args, portfolio_path):
    """
    Execute the main strategy based on the given configuration.

    Args:
        config (StrategyConfig): An object containing the necessary configuration parameters.
    
    Returns:
        None
    """    
    args.present_date = date.today()
    args.past_date = find_business_day(args.present_date, -args.inference_days)
    backtester = Backtester(portfolio_path, args=args, zero=False)   
    action = backtester.strategy.generate_allocation()
    alpaca_execute(action)
    #update portfolio
    backtester.portfolio.execute(action)
    backtester.portfolio.portfolio.to_csv(portfolio_path, index=False) # update portfolio
    


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
    parser.add_argument('--inference_days','-i', type=int, default=20, help='number of days for inference')
    parser.add_argument('--future_days','-f', type=int, default=10, help='number of days in the future to predict')
    parser.add_argument('--newsapi','-n', type=str, default="finhub", help='newsapi: finhub, newsapi, google')
    args = parser.parse_args(raw_args)
    return args
# modes: SimpleStrategy, FinLlama3Strategy, GPTStrategy, ARIMAStrategy 

if __name__ == "__main__":
    args = get_args()
    print(args)
    assert args.strategy != 'ManualStrategy', f"Invalid strategy: {args.strategy}"
    config = StrategyConfig(
        inference_days=args.inference_days,
        future_days=args.future_days,
        strategy=args.strategy,
        newsapi=args.newsapi,
    )
    portfolio_path = f'{ROOT_D}/sd/test_portfolio.csv'
    auto_trade(config, portfolio_path)