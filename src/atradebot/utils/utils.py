# util and helper functions for atradebot

import pandas as pd
from datetime import datetime, timedelta
import yaml
import yfinance as yf
import re
import random
import numpy as np
import json
from atradebot.params import DATE_FORMAT, ALLTICKS, logging, OUTPUT_D
import matplotlib.pyplot as plt
import os

# Print-list of news/LLM answers
def gen_report(backtester, result:pd.DataFrame, fname:str, report:str=""):
    """
    Generate a report by saving a DataFrame to a CSV file.

    Args:
        backtester: The backtester object containing portfolio information.
        result (pd.DataFrame): The DataFrame to be saved as a report.
            contains: 'Date', 'FutureDate', 'Stock', 'Predictions', 'FuturePrice', 'Log': {"prompt", "answer", "news"}
        fname (str): The file name for the CSV file.
        report (str, optional): Additional information to be saved in a text file. Default is an empty string.

    Returns:
        None

    Output:
        Saves the result DataFrame to a CSV file with a specific naming convention.

    Example:
        gen_report(backtester, result, 'report')

    Logging:
        Logs the backtester.portfolio and result information using the logging module.
    """    
    logging.info(backtester.portfolio)
    # Predictions of stocks (rank)
    logging.info(result)
    result.to_csv(f'{OUTPUT_D}/{fname}_data_{str(backtester.args.present_date)}_{backtester.get_strategyName()}.csv', index=False)
    if report:
        with open(f'{OUTPUT_D}/{fname}_report_{str(backtester.args.present_date)}.txt', 'w') as f:
            f.write(report)

def extract_date(filename):
    match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    if match:
        return datetime.strptime(match.group(), '%Y-%m-%d')
    return None

def get_latest_report(dir):
    paths = os.listdir(dir)
    latest_date = None
    latest_files = []
    for path in paths:
        file_date = extract_date(path)
        if file_date:
            if latest_date is None or file_date > latest_date:
                latest_date = file_date
                latest_files = [path]
            elif file_date == latest_date:
                latest_files.append(path)
    return latest_files 

def get_braces(text):
    """
    Extracts the content inside the first pair of braces in a string.

    Args:
        text (str): The input string from which to extract the content inside the braces.

    Returns:
        str: The content inside the first pair of braces found in the input string, including the braces.
    """    
    match = re.search(r'\{(.*)\}', text)  # Match the first occurrence of {...}
    if match:
        return '{'+match.group(1)+'}'  # Extract the content inside the braces
    return None

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


def get_tickers(text: str):
    """
    Retrieve tickers from a given text.

    Args:
        text (str): The input text to search for tickers.

    Returns:
        list: A list of tickers found in the input text.
    """    
    tickers = [t for t in ALLTICKS if is_word_in_text(t, text)]
    return tickers

def load_args(args, path='config.txt'):
    """
    Loads args from previous run.

    Args:
        args (ArgumentParser): Use args = get_args_parser() to get the args.
        path (str, optional): Path to saved config, defaults to 'config.txt'.

    Returns:
        ArgumentParser: Loaded args.
    """    
    args = args.parse_args([])
    with open(path, 'r') as f:
        args.__dict__ = json.load(f)
    return args

def is_business_day(date:datetime.date):
    """
    Check if date is a business day.

    Args:
        date (datetime.date): Date to check.

    Returns:
        bool: If date is a business day.
    """    
    return bool(len(pd.bdate_range(date, date)))

def find_business_day(start_date:datetime.date, num_days:int=0):
    """
    Add or subtract days from start_date and find the next business day.

    Args:
        start_date (datetime.date): Date to start from.
        num_days (int): Number of days to add/subtract.

    Returns:
        datetime.date: Output date.
    """ 
    out_date = start_date + timedelta(days=num_days)
    while not is_business_day(out_date):
        if num_days < 0:
            out_date -= timedelta(days=1)
        else:
            out_date += timedelta(days=1)
        
    return out_date

#add/subtract business days
def business_days(start_date:datetime.date, num_days:int):
    """
    Add or subtract business days from start_date.

    Args:
        start_date (datetime.date): Date to start from.
        num_days (int): Number of days to add/subtract.

    Returns:
        datetime.date: Output date.
    """    
    current_date = start_date
    business_days_added = 0
    while business_days_added < abs(num_days):
        if num_days < 0:
            current_date -= timedelta(days=1)
        else:
            current_date += timedelta(days=1)
        if current_date.weekday() < 5:  # Monday to Friday
            business_days_added += 1
    return current_date


def pd_append(df, dict_d):
    """
    Append dictionary to dataframe.

    Args:
        df (pandas.DataFrame): Pandas dataframe.
        dict_d (dict): Dictionary to append.
    
    Returns:
        pandas.DataFrame: Pandas dataframe with appended dictionary.
    """ 
    return pd.concat([df, pd.DataFrame.from_records([dict_d])])

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

def get_config(cfg_file):
    """
    Reads a YAML configuration file and returns its content as a Python dictionary.

    Args:
        cfg_file (str): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration parameters.
    """    
    with open(cfg_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg

def find_peaks_valleys(array):
    """
    Find peaks and valleys.

    Args:
        array (list): List of numbers.

    Returns:
        list: Reduced list of peaks and valleys. List of ids of the array.

    Raises:
        None
    """    
    peaks = []
    valleys = []
    off = 5
    for i in range(off, len(array) - off, off):
        if array[i - off] < array[i] > array[i + off]:
            peaks.append(i)
        elif array[i - off] > array[i] < array[i + off]:
            valleys.append(i)
    return peaks, valleys


def filter_points(data, peak_idx, valley_idx):
    """
    Ignore peaks and valleys that are too close to each other.

    Args:
        data (pandas): Data from yfinance.download.
        peak_idx (List[int]): List of indices for peaks.
        valley_idx (List[int]): List of indices for valleys.

    Returns:
        List[int]: Reduced list of peaks and valleys.
    """       
    len_min = min(len(peak_idx), len(valley_idx))
    idx = 0
    coef_var = np.std(data)/np.mean(data)
    peak_idx_n, valley_idx_n = [], []
    while idx < len_min:
        abs_diff = abs(data[peak_idx[idx]]-data[valley_idx[idx]])
        min_diff = min(data[peak_idx[idx]], data[valley_idx[idx]])
        percent = abs_diff/min_diff
        if percent > coef_var*0.2: 
            peak_idx_n.append(peak_idx[idx])
            valley_idx_n.append(valley_idx[idx])
        idx += 1
    return peak_idx_n, valley_idx_n
    
# run financial sentiment analysis model
def get_forecast(stock, date, add_days=[21, 5*21, 12*21]):
    """
    Get forecast for stock on date.

    Args:
        stock (str): Stock id.
        date (datetime): Date in datetime format.
        add_days (list, optional): Number of days to add for forecasting in a list, defaults to [21, 5*21, 12*21].

    Returns:
        List[int]: Forecast list based on add_days. E.g., add_days=[21, 5*21, 12*21] return is stock gain/loss in [1mon, 5mon, 1 yr]. Higher than 1 indicates percent increase, lower than 1 indicates percent decrease.
    """
    s = date
    e = business_days(s, +3)
    data = yf.Ticker(stock)
    hdata = data.history(start=s.strftime("%Y-%m-%d"),  end=e.strftime("%Y-%m-%d"))
    price = hdata['Close'].mean()

    forecast = [0, 0, 0]
    for idx, adays in enumerate(add_days): #add business days
        s = business_days(s, adays)#look into future
        e = business_days(s, +3)
        hdata = data.history(start=s.strftime("%Y-%m-%d"),  end=e.strftime("%Y-%m-%d"))
        forecast[idx] = hdata['Close'].mean()/price
    
    return forecast


def get_mentionedtext(keyword, text, context_length=512):
    """
    Get text around keyword given the number of characters to extract 
    before and after the keyword.

    Args:
        keyword (str): Keyword to search for in text.
        text (str): Text to search for keyword.
        context_length (int, optional): Number of words before and after the keyword, defaults to 512.

    Returns:
        str: Collected text around keyword.
    """    
    delimiter = '.'
    # Create a regex pattern to match the keyword with surrounding text
    # pattern = r"(.{0,%d})(%s)(.{0,%d})" % (context_length, re.escape(keyword), context_length)
    pattern = r"([^%s]+%s[^%s]+)%s" % (delimiter, re.escape(keyword), delimiter, delimiter)

    # Find all matches in the text
    matches = re.findall(pattern, text)
    f_text = ''
    for match in matches:
        match = match.replace("\n", "")
        f_text += match
        if len(f_text) > context_length:
            break
    return f_text



def gen_rand_alloc(n_stock=5):
# Generate random percentages that sum up to 100
    """
    Generate random allocation percentages for a list of stocks.

    Args:
        n_stock (int): The number of stocks for which to generate allocation percentages. Default is 5.

    Returns:
        list: A list of random allocation percentages for each stock, summing up to 100.

    Note:
        The last stock's allocation percentage is calculated to ensure that the total sum is 100.
    """    
    percentages = [random.randint(1, 100) for _ in range(n_stock - 1)]
    remaining_percentage = 100 - sum(percentages)
    percentages.append(remaining_percentage)
    return percentages


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