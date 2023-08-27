"""Stocks module for getting stock data"""
from time import time
import yfinance as yf


def current_price(symbol):
    """Get current price"""
    stock = yf.Ticker(symbol)
    return [int(time()), stock.info['currentPrice']]

