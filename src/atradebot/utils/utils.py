import json
import re
from atradebot.config import ALLTICKS, DATE_FORMAT
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time
import os
import matplotlib.pyplot as plt


def portfolio_update(allocation: dict, portfolio): 
    # Apply allocation to portfolio and save in csv file
    for ticker, qty in allocation.items():
        if ticker in portfolio['Ticker'].values:
            portfolio.loc[portfolio['Ticker'] == ticker, 'Quantity'] = qty
        else:
            # If ticker not in portfolio, add it with default values
            price = 0
            info = yf.Ticker(ticker).info
            price = info.get('regularMarketOpen', 0)
            new_row = {
                'Ticker': ticker,
                'Quantity': qty,
                'Price': price,
                'Cost': price * qty,
                'Unit Cost': price,
            }
            portfolio = pd.concat([portfolio, pd.DataFrame([new_row])], ignore_index=True)
    return portfolio

def is_json_in_text(text, keys = None) -> bool:
    matches = list(re.finditer(r'(\{.*\}|\[.*\])', text, re.DOTALL))
    if not matches:
        return False
    
    ret = False
    for match in matches:
        if keys:
            json_str = match.group(1)
            ret = all(key in json_str for key in keys)
        else:
            ret = True
        if ret:
            break
    return ret
    
def find_json(response_text):
    # NO lists of JSONs, only first valid JSON object or array
    # Check if response starts with [ and ends with ] (list of JSONs)
    response_text = response_text.strip()
    if response_text.startswith('[') and response_text.endswith(']'):
        print(f"WARNING: Response is a list of JSONs {response_text}")

    a_json = {}
    try:
        # Find all {...} or [...] blocks using regex
        matches = list(re.finditer(r'(\{.*?\}|\[.*?\])', response_text, re.DOTALL))
        for match in matches:
            json_str = match.group(1).replace("'s", "").replace("'", '"')
            cleaned = re.sub(r'#.*', '', json_str).strip() # Remove inline comments starting with #
            cleaned = cleaned.replace('\n', ' ')
            try:
                a_json = json.loads(cleaned)
                break  # Stop at the first valid JSON
            except json.JSONDecodeError:
                print(f"JSON decode error, trying next match... {cleaned}")
                continue
    except Exception:
        print(f"No JSON found in the response. {response_text}")
        a_json = {}

    return a_json

def find_number_in_text(text):
    match = re.search(r'(-?\d+(?:\.\d+)?)(?=%|\s|$)', text)
    if match:
        return float(match.group(1))
    return 0

def is_word_in_text(word, text):
    # Use regex with word boundaries (\b) for full-word matching
    """
    Check if a word is in a given text.

    Args:
        word (str): The word to search for.
        text (str): The text in which to search for the word.

    Returns:
        bool: True if the word is found in the text, False otherwise.
    """    
    pattern = r'\b{}\b'.format(re.escape(word))
    return bool(re.search(pattern, text))

def get_tickers(text: str):
    """
    Retrieve tickers from a given text.

    Args:
        text (str): The input text to search for tickers.

    Returns:
        list: A list of tickers found in the input text.
    """    
    response = text 
    tickers = [t for t in ALLTICKS if is_word_in_text(t, response)]
    return tickers

def is_business_day(date:date):
    """
    Check if date is a business day.

    Args:
        date (datetime.date): Date to check.

    Returns:
        bool: If date is a business day.
    """    
    return bool(len(pd.bdate_range(date, date)))

def find_business_day(start_date:date, num_days:int=0):
    """
    Add or subtract days from start_date and find the next business day.

    Args:
        start_date (datetime.date): Date to start from.
        num_days (int): Number of days to add/subtract.

    Returns:
        datetime.date: Output date.
    """ 
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, DATE_FORMAT).date()
        
    out_date = start_date + timedelta(days=num_days)
    while not is_business_day(out_date):
        if num_days < 0:
            out_date -= timedelta(days=1)
        else:
            out_date += timedelta(days=1)
        
    return out_date

def get_existing_tickers(filepath):
    existing_stocks = set()
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            stock_ = json.load(f)
        existing_stocks = set(stock_.keys())
        # with open(filepath, "r") as file:
        #     for line in file:
        #         try:
        #             entry = json.loads(line.strip())
        #             if 'ticker' in entry:
        #                 existing_stocks.add(entry['ticker'])
        #         except json.JSONDecodeError:
        #             continue
    return existing_stocks

def timestamp_json(obj):
    if isinstance(obj, (datetime, date, time, pd.Timestamp)):
        return obj.strftime(DATE_FORMAT) # Converts to "YYYY-MM-DDTHH:MM:SS.ffffff" format
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def extract_json_from_response(content: str) -> dict | None:
    """Extracts JSON from markdown-formatted response."""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7 :]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON from response: {e}")
    return None



def plot_stock(stocks:dict, show=False):
    # stocks: {"SPY":data_spy/data_spy[0], #normalize gains "ThisPortfolio":data_my/init_capital}
    # Plotting configuration
    """
    Plot stock prices for each stock in the input dictionary.

    Args:
        stocks (dict): A dictionary where keys are stock names and values are pandas Series with date index.
        show (bool, optional): Whether to display the plot. Default is False.

    Returns:
        matplotlib.pyplot: The plot showing stock price comparison.

    Example:
        plot_stock({'AAPL': apple_stock_data, 'GOOGL': google_stock_data}, show=True)

    Note:
        Requires matplotlib library for plotting.
    """    
    plt.figure(figsize=(10, 6))  # Set the figure size
    for k, stock_data in stocks.items():
        plt.plot(stock_data.index, stock_data.values, label=k)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Comparison')
    plt.legend()
    if show:
        plt.show()
    return plt


def extract_last_number(text):
    """
    Extract the last number from a given text string.

    Args:
        text (str): The input text string containing numeric values.

    Returns:
        float: The last number found in the text string, or None if no numbers are found.
    """    
    matches = re.findall(r'-?\d+\.?\d*', text)
    if matches:
        return float(matches[-1])  # Return the last number as a float
    return None

def get_datetime(date_str):
    """
    Convert a date string to a datetime object.

    Args:
        date_str (str or datetime): A date string in the format specified by DATE_FORMAT or a datetime object.

    Returns:
        datetime.date: The date extracted from the input date string.
    """    
    if isinstance(date_str, str):
        return datetime.strptime(date_str, DATE_FORMAT).date()
    return date_str


