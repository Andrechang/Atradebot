import asyncio
from datetime import datetime
from atradebot.utils.utils import get_tickers, gen_report
from atradebot.utils.backtest import Backtester  
from atradebot.data.yf_data import get_price
from atradebot.data.get_news import run_gptresearch
from atradebot.strategies.strategy import StrategyConfig
from argparse import ArgumentParser
from atradebot.strategies import __MODEL_REGISTRY__
from datetime import date, datetime
from atradebot.params import DATE_FORMAT, ROOT_D
from atradebot.utils.utils import find_business_day, gen_report
from atradebot.utils.alpaca_trade import alpaca_execute
import os

MARKET_CAP = 'large or medium cap'
SECTOR = 'information technology'

# findstocks: function to search for stock ideas to add to a portfolio
def findstocks(args:StrategyConfig, verbose:bool=False, sell:bool=False, skip_tickers:list=None):
    """
    Find the stocks based on the strategy configuration provided.

    Args:
        args (StrategyConfig): An object that contains the strategy configuration.
        verbose (bool, optional): Whether to print verbose output. Default is False.
        sell (bool, optional): Whether to find stocks to sell. Default is False.
        skip_tickers (list, optional): A list of stock tickers to skip. Default is None.
    Returns:
        None

    Prints:
        If verbose is True, it prints the answer, stock list, and prediction report.

    Raises:
        None
    """    
    stklist, answer = stock_research(sell, skip_tickers)
    portfolio_path = os.path.join(ROOT_D,'tmp.csv')
    with open(portfolio_path, 'w') as f:
        f.write('Description,Ticker,Quantity,Price,Cost,Unit Cost,Value,Gain,Gain %\n')
        for stk in stklist:
            price = get_price(stk)
            f.write(f'{stk},{stk},1,{price},{price},{price},0,0,0\n')

    if verbose:
        print(answer)
        print(stklist)
        if sell:
            print('SELL PREDICTION REPORT'+20*'=')
        else: 
            print('BUY PREDICTION REPORT'+20*'=')
    # run backtest to evaluate the portfolio-----------------------
    backtester = Backtester(portfolio_path, args=args, zero=True)    
    future_days = [30, 120, 360]
    pred_result = backtester.rank_predict(args.present_date, future_days)
    if sell:
        gen_report(backtester, pred_result, 'findsell', report=answer)
    else:
        gen_report(backtester, pred_result, 'find', report=answer)
    
    if args.run_trade and sell==False:
        action = backtester.strategy.generate_allocation()
        alpaca_execute(action)

    return backtester
   
def stock_research(sell=False, skip_tickers:list=None):
    """
    Perform stock research and provide a list of the best 10 growth stocks in the information technology sector.
    Args:
        sell (bool): If True, search for stocks to sell. Default is False.
        skip_tickers (list): List of stock tickers to skip. Default is None.
    Returns:
        tuple: A tuple containing the list of tickers for the recommended stocks and the research answer.

    Raises:
        Any exceptions that may occur during the execution.
    """    
    cur_date = datetime.now().strftime(DATE_FORMAT)
    # get news to search for stocks
    if sell:
        sysPrompt = 'You are a financial advisor. Given today is {}, ' \
            'respond with a list of the worst 10 stocks in the {} sector.' \
            'List {} stocks that is expected to decline their stock price in a year in the future.'
        if skip_tickers is not None:
            skip_ = ', '.join(skip_tickers)
            sysPrompt += f' In your analysis, avoid the following stocks: {skip_}'
    else:
        sysPrompt = 'You are a financial advisor. Given today is {}, ' \
            'respond with a list of the best 10 growth stocks in the {} sector.' \
            'List {} stocks that is expected to grow their stock price in a year in the future.'
        
    query = sysPrompt.format(cur_date, SECTOR, MARKET_CAP)
    answer = asyncio.run(run_gptresearch(query))
    p = get_tickers(answer)
    return p, answer


def get_args(raw_args=None):
    """
    Parse and return the command-line arguments.

    Args:
        raw_args (list): A list of strings representing the command-line arguments. Default is None.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.

    Description:
        This function uses the ArgumentParser class to define and parse command-line arguments. The function accepts the following arguments:
        --strategy (-s): The strategy to use for prediction. Default is 'GPTStrategy'.
        --inference_days (-i): Number of days for inference. Default is 20.
        --future_days (-f): Number of days in the future to predict. Default is 10.
        --newsapi (-n): The source of news articles. Options are 'finhub', 'newsapi', or 'google'. Default is 'finhub'.

    Example:
        To parse command-line arguments:
        ```
        args = get_args()
        print(args.strategy)
        print(args.inference_days)
        print(args.future_days)
        print(args.newsapi)
        ```
    """    
    parser = ArgumentParser(description="parameters")
    parser.add_argument('--strategy','-s', type=str, default='GPTStrategy', help=f'Modes: {__MODEL_REGISTRY__.keys()}')
    parser.add_argument('--inference_days','-i', type=int, default=20, help='number of days for inference')
    parser.add_argument('--future_days','-f', type=int, default=10, help='number of days in the future to predict')
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
    config.present_date = date.today()
    config.past_date = find_business_day(config.present_date, -config.inference_days)
    findstocks(config)
    

    
    
    

    