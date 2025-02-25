# get historical OHLCV data from yfinance
import pandas as pd
import os
from atradebot.params import YFCACHE, FHUB_API
import yfinance as yf
import finnhub
from tenacity import retry, stop_after_attempt, wait_fixed

yf.set_tz_cache_location(f"{YFCACHE}/.yfcache")

finnhub_client = finnhub.Client(api_key=FHUB_API)

YFDETAIL_KEYS = ['beta', 'forwardPE', 'volume', 'regularMarketVolume', 'averageVolume', 'averageVolume10days', 'averageDailyVolume10Day',
            'marketCap', 'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'priceToSalesTrailing12Months', 'fiftyDayAverage', 'twoHundredDayAverage', 'enterpriseValue', 'profitMargins',
            'floatShares', 'sharesOutstanding', 'sharesShort', 'sharesShortPriorMonth','heldPercentInsiders', 'heldPercentInstitutions', 'shortRatio','impliedSharesOutstanding', 'bookValue', 'trailingEps', 'forwardEps','52WeekChange', 'targetHighPrice', 'targetLowPrice', 'targetMeanPrice',
            'targetMedianPrice', 'enterpriseToRevenue', 'enterpriseToEbitda','totalCash', 'totalCashPerShare', 'ebitda', 'totalDebt', 'quickRatio','currentRatio', 'totalRevenue', 'revenuePerShare', 'returnOnAssets',
            'grossProfits', 'freeCashflow', 'operatingCashflow', 'revenueGrowth','grossMargins', 'ebitdaMargins', 'operatingMargins']


def get_yf_detail(ticker:str) -> str:
    company = yf.Ticker(ticker)
    longBusinessSummary = company.info.get('longBusinessSummary', 'unknown')
    industry = company.info.get('industry', 'unknown')
    country = company.info.get('country', 'unknown')
    company_text = f"Description: {longBusinessSummary}, Industry: {industry}, Location: {country}"
    officers_text = ""
    for officer in company.info['companyOfficers']:
        officers_text += f"{officer['name']}, {officer['title']} \n"
    fullTimeEmployees = company.info.get('fullTimeEmployees', 'unknown')
    company_person = f"Number of employees: {fullTimeEmployees}, Officers: {officers_text} \n"
    sc = [f"{k}: {v}" for k, v in company.info.items() if k in YFDETAIL_KEYS]
    company_details = ', '.join(sc)
    
    return company.info, company_text, company_person, company_details

def get_yf_data_all(ticker:str) -> pd.DataFrame:
    """
    Get all data from yfinance for a stock, with a fallback to Polygon.

    Args:
        ticker (str): Stock ticker.

    Returns:
        pd.DataFrame: Data for the stock.
    """    
    try:
        # Try to get data from yfinance
        data = yf.download(ticker, period="max", interval="1d")
    except Exception as e:
        print(f"yfinance data retrieval failed: {e}.")
    return data

def get_yf_data_some(ticker:str, past_date:str, present_date:str) -> pd.DataFrame:
    """
    Retrieve Yahoo Finance data for a specific ticker within a date range, with a fallback to Polygon.

    Args:
        ticker (str): The ticker symbol of the stock.
        past_date (str): The start date in YYYY-MM-DD format.
        present_date (str): The end date in YYYY-MM-DD format.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the stock data.
    """    
    try:
        # Try to get data from yfinance
        data = yf.download(ticker, start=past_date, end=present_date, interval="1d")
    except Exception as e:
        print(f"yfinance data retrieval failed: {e}.")
    return data

def get_price(stock):
    """
    Get current price of stock.

    Args:
        stock (str): Stock id.

    Returns:
        float: Current price.
    """    
    ticker = yf.Ticker(stock).info
    return ticker['regularMarketOpen']

def get_price_date(date, present_date, stock):
    """
    Get the price of a stock for a given date range.

    Args:
        date (str): Start date to get the stock price.
        present_date (str): End date to get the stock price.
        stock (str): Stock ID.
    
    Returns:
        pandas.DataFrame: Dataframe of stock data from the date to the present_date.
    """    
    data = yf.Ticker(stock)
    hdata = data.history(start=date,  end=present_date)
    return hdata

def get_finhub_data(ticker: str) -> dict:
    """
    Get stock data from Finnhub.

    Args:
        ticker (str): Stock ticker.

    Returns:
        dict: Data for the stock.
    """
    data = finnhub_client.company_profile2(symbol=ticker)
    return data

def get_finhub_price(ticker: str) -> float:
    """
    Get current price of stock from Finnhub.

    Args:
        ticker (str): Stock ticker.

    Returns:
        float: Current price.
    """
    quote = finnhub_client.quote(symbol=ticker)
    return quote['c']
