# util and helper functions for atradebot

import pandas as pd
from datetime import datetime, timedelta
import yaml
import yfinance as yf

DATE_FORMAT = "%Y-%m-%d"


def is_business_day(date):
    return bool(len(pd.bdate_range(date, date)))


#add/subtract business days
def business_days(start_date, num_days):
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
    return pd.concat([df, pd.DataFrame.from_records([dict_d])])


def get_price(stock):
    ticker = yf.Ticker(stock).info
    return ticker['regularMarketOpen']


# holdings: Name, Qnt, UCost (unit cost), BaseCost, Price (current price), Value (current Value), LongGain (Qnt), ShortGain (Qnt)
# activity: Name, type (buy/sell), TB (time bought), Qnt, Proceed
# balance: Time, Cash, Stock, Total
# news: Time, Name, Text, Score, Link
def get_config(cfg_file):
    with open(cfg_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


# run financial sentiment analysis model
def get_forecast(stock, date):
    """get forecast for stock on date
    Args:
        stock (string): stock id
        date (string): date format 'Feb 2, 2019' "%b %d, %Y"

    Returns:
        forecast: forecast list [1mon, 5mon, 1 yr], higher than 1. percent increase, lower than 1. percent decrease
    """    
    s = datetime.strptime(date, "%b %d, %Y")
    e = business_days(s, +3)
    data = yf.Ticker(stock)
    hdata = data.history(start=s.strftime("%Y-%m-%d"),  end=e.strftime("%Y-%m-%d"))
    price = hdata['Close'].mean()

    forecast = [0, 0, 0]
    add_days = [21, 5*21, 12*21] #add business days
    for idx, adays in enumerate(add_days):
        s = business_days(s, adays)#look into future
        e = business_days(s, +3)
        hdata = data.history(start=s.strftime("%Y-%m-%d"),  end=e.strftime("%Y-%m-%d"))
        forecast[idx] = hdata['Close'].mean()/price
    
    return forecast