# place for API calls and interactions
# MCP: https://github.com/alpacahq/alpaca-mcp-server

# Test0:small cap tech stocks only
# PK27NSU95PPVGB1HRITW
# WE3MqRpptElNEzd0qgjQ3V7iqOzs0wXUM2ULCU4z

# Test1: large+small cap tech stocks
# PK2G4IISOE6CG4HR8U4B
# pfXcKjk45Yy1W3LN56IupTqCcXR4VGaTI1HM4Z20

# Option test:
# PK377NR2ITAAR7MQWAUS
# YdhUzWRDDKunw0H5fyHTD1J6Jh1vJoCCUV3EEQWD

from atradebot.config import *
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.client import TradingClient
from datetime import datetime, timedelta
import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoLatestQuoteRequest


# integration with alpaca API
class AlpacaAPI():
    def __init__(self, mode='test'):
        try:
            if mode == 'test2': 
                self.client = TradingClient(ALPACA_API2, ALPACA_SECRET2, paper=PAPERTRADE)
            else: # test or prod
                self.client = TradingClient(ALPACA_API, ALPACA_SECRET, paper=PAPERTRADE)
        except Exception as e:
            print(f"An error occurred while initializing Alpaca: {e}")

    def _buy(self, stock, allocation):
        market_order_data = MarketOrderRequest(
                            symbol=stock,
                            side=OrderSide.BUY,
                            qty=allocation,
                            time_in_force=TimeInForce.DAY,
                            )
        market_order = self.client.submit_order(order_data=market_order_data)
        return market_order

    def _sell(self, stock, allocation):
        market_order_data = MarketOrderRequest(
                            symbol=stock,
                            side=OrderSide.SELL,
                            qty=allocation,
                            time_in_force=TimeInForce.DAY,
                            )
        market_order = self.client.submit_order(order_data=market_order_data)
        return market_order

    def alpaca_execute(self, allocation):
        """
        Execute trades based on the given allocation using Alpaca API.
        Args:
            allocation (dict): A dictionary containing stock tickers as keys and their respective allocations as values.
        Returns:
            None    
        """
        for stock, allocation in allocation.items():
            if allocation > 0:
                self._buy(stock, allocation)
            elif allocation < 0:
                self._sell(stock, -allocation)

    def alpaca_get_portfolio(self):
        """
        Get the current portfolio from Alpaca.

        Returns:
            Portfolio: An object representing the portfolio containing stock tickers and their respective allocations.
        """
        positions = self.client.get_all_positions()
        dict_port = {}
        for position in positions:
            if position.qty != 0:
                dict_port[position.symbol] = (float(position.qty), float(position.cost_basis), float(position.current_price)) # tuple (quantity, cost basis, current price)
        portfolio = []
        for ticker, (quantity, cost_basis, cur_price) in dict_port.items():
            unit_cost = cost_basis / quantity if quantity != 0 else 0
            portfolio.append({
                "Ticker": ticker,
                "Quantity": quantity,
                "Cost": cost_basis,
                "Unit Cost": unit_cost,
                "Price": cur_price
            })
        df = pd.DataFrame(portfolio)
        if df.empty:
            df = pd.DataFrame(columns=["Ticker", "Quantity", "Cost", "Unit Cost", "Price"])
            print("Alpaca Portfolio is empty. Created an empty portfolio file.")

        # Get account information
        account = self.client.get_account()
        available_cash = account.cash

        return df, available_cash

    # Options ========================================
    # level1
    # - Sell a covered call - User must own sufficient underlying shares
    # - Sell cash-secured put - User must have sufficient options buying power

    # level2 User must have sufficient options buying power
    # - Buy a call
    # - Buy a put

    # level3 User must have sufficient options buying power
    # - Buy a call spread
    # - Buy a put spread

    # Ensuring qty is a whole number
    # Notional must not be populated
    # time_in_force must be day
    # extended_hours must be false or not populated
    # type must be marketor limit



    def alpaca_execute_option(self, allocation):

        for option, params in allocation.items():
            if params['qty'] > 0:
                self.buy_option(option, **params)
            elif params['qty'] < 0:
                self.sell_option(option, **params)

    def buy_option(self, symbol, qty, limit_price, expiry_date, strike_price, option_type):
        """
        Buy an option using Alpaca API.

        Args:
            symbol (str): The stock symbol for the option.
            qty (int): Quantity of contracts to buy.
            limit_price (float): The limit price for the option.
            expiry_date (str): Expiry date of the option in 'YYYY-MM-DD' format.
            strike_price (float): Strike price of the option.
            option_type (str): Type of the option ('call' or 'put').

        Returns:
            dict: Response from Alpaca API.
        """
        option_order_data = LimitOrderRequest(
            symbol=symbol,
            side=OrderSide.BUY,
            qty=qty,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
            option_type=option_type,
            expiry_date=expiry_date,
            strike_price=strike_price
        )
        return self.client.submit_order(order_data=option_order_data)

    def sell_option(self, symbol, qty, limit_price, expiry_date, strike_price, option_type):
        """
        Sell an option using Alpaca API.

        Args:
            symbol (str): The stock symbol for the option.
            qty (int): Quantity of contracts to sell.
            limit_price (float): The limit price for the option.
            expiry_date (str): Expiry date of the option in 'YYYY-MM-DD' format.
            strike_price (float): Strike price of the option.
            option_type (str): Type of the option ('call' or 'put').

        Returns:
            dict: Response from Alpaca API.
        """
        option_order_data = LimitOrderRequest(
            symbol=symbol,
            side=OrderSide.SELL,
            qty=qty,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
            option_type=option_type,
            expiry_date=expiry_date,
            strike_price=strike_price
        )
        return self.client.submit_order(order_data=option_order_data)



# Crypto ========================================

def get_btc_price():
    """Fetches and prints the latest bid/ask price for BTC/USD from Alpaca."""
    try:
        # The CryptoHistoricalDataClient does not require API keys.
        client = CryptoHistoricalDataClient()

        # Define the request to get the latest quote for BTC/USD.
        request_params = CryptoLatestQuoteRequest(symbol_or_symbols="BTC/USD")

        # Get the latest quote data.
        latest_quote = client.get_crypto_latest_quote(request_params)

        # The response is a dictionary with the symbol as the key.
        btc_quote = latest_quote["BTC/USD"]

        # The "current price" is not a single value in trading; it is the bid/ask spread.
        # The ask price is the lowest price a seller is willing to accept.
        ask_price = btc_quote.ask_price
        # The bid price is the highest price a buyer is willing to pay.
        bid_price = btc_quote.bid_price

        print("Latest Bitcoin Price (BTC/USD):")
        print(f"  Bid: ${bid_price}")
        print(f"  Ask: ${ask_price}")

    except Exception as e:
        print(f"An error occurred: {e}")




