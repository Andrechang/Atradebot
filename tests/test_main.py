# regression tests

import pytest
from atradebot import main
from atradebot import backtest
import yfinance as yf

@pytest.mark.test0
def test_get_profile():
    _, config = main.get_arg([])
    bot = main.TradingBot(config)
    assert not bot.holdings.empty

@pytest.mark.test0
def test_get_news():
    _, config = main.get_arg([])
    bot = main.TradingBot(config)
    bot.get_news()
    assert not bot.news.empty

@pytest.mark.test0
def test_backtest():
    # Example usage
    past_date = "2018-01-31" 
    start_date = "2019-01-31"  
    end_date = "2020-05-20" 
    stocks = ['AAPL','ABBV','AMZN','MSFT','NVDA','TSLA', 'SPY', 'VUG', 'VOO']
    data = yf.download(stocks, start=past_date, end=end_date)
    INIT_CAPITAL = 10000
    # Create a portfolio backtester instance
    backtester = backtest.PortfolioBacktester(initial_capital=INIT_CAPITAL, data=data, stocks=stocks, start_date=start_date)
    strategy = backtest.SimpleStrategy(start_date, end_date, data, stocks, INIT_CAPITAL)
    # Run the backtest using the simple strategy
    backtester.run_backtest(strategy)
    # Retrieve the portfolio value
    portfolio_value = backtester.get_portfolio_value()

    assert not portfolio_value.empty

