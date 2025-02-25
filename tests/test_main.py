# regression tests

import pytest
from atradebot.data.yf_data import get_yf_data_all, get_yf_data_some
from atradebot.data.get_news import get_news
from atradebot.utils.portfolio import Portfolio
from atradebot.utils.utils import get_datetime, find_business_day
from atradebot.utils.backtest import backtest
from datetime import datetime, date
from atradebot.params import DATE_FORMAT
from atradebot.utils.search import stock_research, findstocks
from atradebot.strategies.strategy import StrategyConfig
from atradebot.main import manage
from atradebot.params import ROOT_D

@pytest.mark.test
def test_portfolio():
    portfolio_path = f'{ROOT_D}/sd/test_portfolio.csv'
    args = StrategyConfig(
        past_date='2023-08-01',
        present_date='2023-08-22',
        inference_days=5,
        future_days=5,
        strategy='SimpleStrategy'
    )
    portfolio = Portfolio(portfolio_path, args)
    beta, alpha = portfolio.calc_beta()
    print("The portfolio beta is", round(beta, 4))
    print("The portfolio alpha is", round(alpha,5))

    bdays = 3
    sloss = 0.4
    print(f"The stocks going down past {bdays} days: ", portfolio.stop_down(bdays))
    print(f"The stocks went down {sloss} from cost basis are: ", portfolio.stop_loss(loss=sloss))

    portfolio.update_portfolio(day=get_datetime('2023-08-08'))
    print(portfolio)
    portfolio.buy_stock('AAPL', 10)
    print(portfolio)
    portfolio.sell_stock('AAPL', 5)
    print(portfolio)
    #buy new stock
    portfolio.buy_stock('TSLA', 10)
    print(portfolio)
    portfolio.sell_stock('TSLA', 5)
    print(portfolio)

@pytest.mark.test
def test_get_yfdata():
    data = get_yf_data_some("AAPL", "2024-10-15", "2024-10-17")
    assert data is not None

@pytest.mark.test
def test_search():
    stocks, answ = stock_research()

@pytest.mark.test
def test_getnews():
    news = get_news(stock='A', num_results=10, time_period=[], news_source='dataset')

# main apps
@pytest.mark.apps
def test_findstocks():
    inference_days = 5
    present_date = date.today()
    past_date = find_business_day(present_date, -inference_days)
    args = StrategyConfig(
        past_date=past_date,
        present_date=present_date,
        inference_days=inference_days,
        future_days=10, 
        strategy='FinLlama3Strategy'
    )
    findstocks(args)

@pytest.mark.apps
def test_managestocks():
    inference_days = 5
    present_date = date.today()
    past_date = find_business_day(present_date, -inference_days)
    args = StrategyConfig(
        past_date=past_date,
        present_date=present_date,
        inference_days=inference_days,
        future_days=10, 
        strategy='FinLlama3Strategy'
    )
    manage(args, portfolio_path=f'{ROOT_D}/sd/test_portfolio.csv')
    

TEST_PAST_DATE = datetime.strptime('2023-08-01', DATE_FORMAT).date()
TEST_PRESENT_DATE = datetime.strptime('2023-08-22', DATE_FORMAT).date() 
# model tests
@pytest.mark.backtest
def test_backtest_alloc_simple():
    past_date = TEST_PAST_DATE
    present_date = TEST_PRESENT_DATE
    args = StrategyConfig(
        past_date=past_date,
        present_date=present_date,
        inference_days=10,
        future_days=10, 
        strategy='SimpleStrategy'
    )
    backtest(
        portfolio_path=f'{ROOT_D}/sd/test_portfolio.csv',
        args=args,
        eval_mode='alloc',
        plot=False)

@pytest.mark.backtest
def test_backtest_predict_simple():
    past_date = TEST_PAST_DATE
    present_date = TEST_PRESENT_DATE
    args = StrategyConfig(
        past_date=past_date,
        present_date=present_date,
        inference_days=5,
        future_days=5, 
        strategy='SimpleStrategy'
    )
    backtest(
        portfolio_path=f'{ROOT_D}/sd/test_portfolio.csv',
        args=args,
        eval_mode='predict',
        plot=False)

    
    
  
