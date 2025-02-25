# integration with alpaca API

from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.client import TradingClient
import pandas as pd
from atradebot.params import ALPACA_API, ALPACA_SECRET, PAPERTRADE
from atradebot.utils.portfolio import Portfolio, dict2portfolio
from atradebot.utils.utils import find_business_day
from datetime import date
from atradebot.strategies.strategy import StrategyConfig

ALPACA_CLIENT = TradingClient(ALPACA_API,ALPACA_SECRET,paper=PAPERTRADE)

def _buy(stock, allocation):
    market_order_data = MarketOrderRequest(
                        symbol=stock,
                        side=OrderSide.BUY,
                        qty=allocation,
                        time_in_force=TimeInForce.DAY,
                        )
    market_order = ALPACA_CLIENT.submit_order(order_data=market_order_data)
    return market_order

def _sell(stock, allocation):
    market_order_data = MarketOrderRequest(
                        symbol=stock,
                        side=OrderSide.SELL,
                        qty=allocation,
                        time_in_force=TimeInForce.DAY,
                        )
    market_order = ALPACA_CLIENT.submit_order(order_data=market_order_data)
    return market_order

def alpaca_execute(allocation):
    """
    Execute trades based on the given allocation using Alpaca API.
    Args:
        allocation (dict): A dictionary containing stock tickers as keys and their respective allocations as values.
    Returns:
        None    
    """
    for stock, allocation in allocation.items():
        if allocation > 0:
            _buy(stock, allocation)
        elif allocation < 0:
            _sell(stock, -allocation)

def alpaca_get_portfolio():
    """
    Get the current portfolio from Alpaca.

    Returns:
        Portfolio: An object representing the portfolio containing stock tickers and their respective allocations.
    """
    positions = ALPACA_CLIENT.get_all_positions()
    dict_port = {}
    for position in positions:
        if position.qty != 0:
            dict_port[position.symbol] = (float(position.qty), float(position.cost_basis)) # tuple (quantity, cost basis)

    portfolio_path = dict2portfolio()
    present_date = date.today()
    past_date = find_business_day(present_date, -30)
    args = StrategyConfig(
        past_date=past_date,
        present_date=present_date,
    )
    portfolio = Portfolio(portfolio_path, args)
    return portfolio