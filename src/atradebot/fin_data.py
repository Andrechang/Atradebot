# create dataset for training a financial model on Yahoo Finance data API, Google News

import os
from argparse import ArgumentParser
import pandas as pd
import yfinance as yf
import numpy as np
from atradebot import main, news_utils, utils
from datasets import Dataset
import time
import json
from datasets import load_dataset, Dataset
from tqdm import tqdm
from datetime import datetime

def collect_events(stock, start_date, end_date, ret=False):
    """collect events to gather news
    
    :param stock: stock id
    :type stock: str
    :param start_date: date start to collect data in format yyyy-mm-dd
    :type start_date: str
    :param end_date: date end to collect data in format yyyy-mm-dd
    :type end_date: str
    :param ret: if True, return random events, defaults to False
    :type ret: bool, optional
    :return: list of dates to collect news
    :rtype: list[pandas.Timestamp]
    """    
    
    data = yf.download(stock, start=start_date, end=end_date)
    if data.empty or ret: # generate random events if no data
        l = data.index.to_list()
        events = [ll for i, ll in enumerate(l) if i%2]
        return events

    data_mean = []
    for i in range(len(data['Close'])-10):
        data_mean.append(data['Close'][i:i+10].mean())

    # collect when to gather news
    p, v = utils.find_peaks_valleys(data_mean)
    # index of data_mean that are peaks and valleys
    peak_idx, valley_idx = utils.filter_points(data_mean, p, v) 
    events = peak_idx + valley_idx #concat lists
    events += [2] # add second day as first event to collect news
    events.sort()
    events_dates = [data.index[event] for event in events]
    return events_dates


def gen_news_dataset(stocks, start_date, end_date, num_news=5, sample_mode = 'sp500', news_source = 'finhub'):
    """collect news at specific dates for stocks
    GoogleSearch API only allows 100 requests per day
    finhub only up to 1 year old news and 60 requests per minute

    :param stocks: list of stock ids
    :type stocks: list[str]
    :param start_date: date start to collect data in format yyyy-mm-dd
    :type start_date: str
    :param end_date:  date end to collect data in format yyyy-mm-dd
    :type end_date: str
    :param num_news: number of news to collect, defaults to 5
    :type num_news: int, optional
    :param sample_mode: 'sp500', sample or 'stocks' sample: collect news in random dates
            stocks: collect news based on up and down changes for each stock
            sp500: collect news based on up and down changes for SP500, defaults to 'sp500'
    :type sample_mode: str, optional
    :param news_source: 'google' or 'finhub', defaults to 'finhub'
    :type news_source: str, optional
    :return: dataset in hugginface format
    :rtype: huggingface dataset
    """    
    assert sample_mode in ['sp500', 'stocks', 'samples']
    assert news_source in ['google', 'finhub']

    all_news = []

    done = [] #keep track of stocks already done

    if sample_mode == 'samples':
        events = collect_events('SPY', start_date, end_date, ret=True)
    elif sample_mode == 'sp500':
        events = collect_events('SPY', start_date, end_date)

    for stock in tqdm(stocks):
        if stock in done:
            continue
        if sample_mode == 'stocks':
            events = collect_events(stock, start_date, end_date)

        print(f'{stock}, events {len(events)}')
        for event in events:
            start = main.business_days(event, -1)#one day before
            start = start.strftime(main.DATE_FORMAT)
            end = main.business_days(event, +1)#one day after
            end = end.strftime(main.DATE_FORMAT)
            news = news_utils.get_news(stock, [start, end], num_news, news_source)
            if not news:
                continue
            all_news += news
        time.sleep(5)

    dataset = Dataset.from_list(all_news)
    return dataset


def generate_forecast_task(data): 
    """
    generate dataset task to train a instruction to forecast model
    also saves json dataset for lit-llama alpaca format

    input: 
            ## Instruction: what is the forecast for ... 
            ## Input: date, news title, news snippet
    output:
            ## Response: forecast percentage change for 1mon, 5mon, 1 yr
    Args:
        data (huggingface dataset): dataset in hugginface format for raw news data

    Returns:
        huggingface dataset: dataset in hugginface format for forecast task
    """    
    file_data = []
    for sample in data:
        if isinstance(sample['date'], str):
            s = datetime.strptime(sample['date'], "%b %d, %Y")
        else:
            s = sample['date']
        forecast = utils.get_forecast(sample['stock'], s)
        file_data.append({'instruction': f"what is the forecast for {sample['stock']}", 
                        'input':f"{sample['date']} {sample['title']} {sample['snippet']}", 
                        'output':f"{round(forecast[0], 2)}, {round(forecast[1],2)}, {round(forecast[2],2)}"})   
    dataset = Dataset.from_pandas(pd.DataFrame(data=file_data))
    with open("stock_forecast.json", "w") as outfile:
        json.dump(file_data, outfile)
    return dataset


def generate_allocation_task(data): 
    """
    generate dataset task to train a instruction to forecast model
    also saves json dataset for lit-llama alpaca format

    input: 
            ## Instruction: What is the allocation suggestion given the news for ... 
            ## Input: date, news snippet
    output:
            ## Response: allocation suggestion in percentage

    Args:
        data (huggingface dataset): dataset in hugginface format for raw news data

    Returns:
        huggingface dataset: dataset in hugginface format for allocation task
    """    
    news_date = []
    stocks = []
    prev_date = data['train'][0]['date']
    file_data = []
    for sample in data['train']:
        if sample['date'].date() == prev_date.date():
            news_date.append(sample)
            stocks.append(sample['stock'])
        else:
            # choose collection of stocks news
            num_news = 3
            for i in range(0, len(stocks), num_news):
                stocks_pick = stocks[i:i+num_news]
                news_pick = news_date[i:i+num_news]
                #get forecast and apply simple buy/sell strategy
                rnd_alloc = utils.gen_rand_alloc(len(stocks_pick))
                r_alloc = {}
                for st, rr in zip(stocks_pick, rnd_alloc):
                    r_alloc[st] = int(rr)
                txt = ''
                for s, n in zip(stocks_pick, news_pick):
                    txt += utils.get_mentionedtext(s, n['text'], context_length=128)

                #generate output based on allocation
                file_data.append({
                    'instruction': f"What is the allocation suggestion given the news for {stocks_pick}", 
                    'input':f"{sample['date']} {txt}", 
                    'output':f"{r_alloc}"})
            prev_date = sample['date']
            news_date = []
            stocks = []

    dataset = Dataset.from_pandas(pd.DataFrame(data=file_data))
    with open("stock_alloc.json", "w") as outfile:
        json.dump(file_data, outfile)
    return dataset


def generate_onestock_task(data, num_news = 3, portifolio_scenarios = 10, cash = 10000): 
    """
    generate dataset task to train a instruction to forecast model for one stock
    also saves json dataset for lit-llama alpaca format

    input: 
            ## Instruction: I have {n} {AAPL} stocks and {x} cash to invest. Given the recent news, should I buy, sell or hold {AAPL} stocks ? 
            ## Input: date, news snippet
    output:
            ## Response: allocation suggestion

    Args:
        data (huggingface dataset): dataset in hugginface format for raw news data

    Returns:
        huggingface dataset: dataset in hugginface format for allocation task
    """    
    news_date = []
    stock_id = data['train'][0]['stock'] #analysis of one stock only
    corp_info = yf.Ticker(stock_id).info
    prev_date = data['train'][0]['date']
    invest_interval = 21 #check invest every week days
    file_data = []
    for sample in data['train']:
        if sample['date'].date() == prev_date.date(): 
            # collect news for same day
            news_date.append(sample)
        else:
            # combine news for same day and generate a prompt
            stock_price = utils.get_price_date(sample['date'], utils.business_days(sample['date'], +3), stock_id)
            stock_price = stock_price['Close'].mean()

            for i in range(0, len(news_date), num_news):
                news_pick = news_date[i:i+num_news]

                #create a random scenario portfolio
                scenarios = set()
                for n in range(1, portifolio_scenarios): # create N random portifolio scenarios
                    max_stocks = int(cash/stock_price)
                    stocks_own = np.random.randint(3, max_stocks)
                    stocks_sell = np.random.randint(1, stocks_own)
                    stocks_buy = np.random.randint(1, max_stocks)
                    if (stocks_own, stocks_buy, stocks_sell) in scenarios: # prevent repeated scenarios
                        continue
                    scenarios.add((stocks_own, stocks_buy, stocks_sell))

                    #get forecast and apply simple buy/sell strategy
                    forecast = utils.get_forecast(stock_id, sample['date'], add_days=[invest_interval]) #forecast [1mon]
                    if forecast[0] > 1:
                        r_alloc = f"buy {stocks_buy} {stock_id} stocks"
                    elif forecast[0] < 1: 
                        r_alloc = f"sell {stocks_sell} {stock_id} stocks"
                    else:
                        r_alloc = 'hold'
                    
                    txt = '' # collect only parts of news that references the stock
                    query = f"{stock_id} {corp_info['longName']}"
                    for new in news_pick:
                        txt += utils.get_doc2vectext(query, new['text'])
            
                    #generate output based on allocation
                    file_data.append({
                        'instruction': f"I have {stocks_own} {stock_id} stocks and {cash} cash to invest. \
                            {stock_id} price now is {stock_price} and given the recent news, should I buy, sell or hold {stock_id} stocks ? ", 
                        'input':f"News from {sample['date']}, {txt}", 
                        'output':f"{r_alloc}"})
            prev_date = sample['date']
            news_date = []

    dataset = Dataset.from_pandas(pd.DataFrame(data=file_data))
    with open("stock_onestock.json", "w") as outfile:
        json.dump(file_data, outfile)
    return dataset



def get_arg(raw_args=None):
    parser = ArgumentParser(description="parameters")
    parser.add_argument('-m', '--mode', type=str,
                        default='', help='Create dataset for task modes: forecast | allocation ')
    parser.add_argument('-t', '--thub', type=str,
                        default='', help='push to hub folder name for task dataset')
    parser.add_argument('-r', '--rhub', type=str,
                        default='', help='push to hub folder name for raw news data')
    parser.add_argument('--start_date', type=str, default="2022-08-31", help='start date for trading analysis')
    parser.add_argument('--end_date', type=str, default="2023-06-20", help='end data for trading analysis')
    parser.add_argument('--stocks', type=str, default="AAPL AMZN MSFT NVDA TSLA", help='stocks to analize')
    parser.add_argument('--datasrc', type=str, default="finhub", help='data src finhub or google')

    args = parser.parse_args(raw_args)
    return args


if __name__ == "__main__":
    args = get_arg()
    stocks = args.stocks.split()
    if args.stocks == 'sp500':
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        stocks = sp500.Symbol.to_list()

    dataset = load_dataset('achang/stock_nvda')
    data_task = generate_onestock_task(dataset)
    data_task.push_to_hub('achang/stocks_one_nvda_v2')
    exit(1)


    #collect news data
    dataset = gen_news_dataset(stocks, args.start_date, args.end_date, 
                                sample_mode='sp500', news_source=args.datasrc, num_news=20)
    if args.rhub != '': #save raw news data
        dataset.push_to_hub(args.rhub)

    if args.mode == 'forecast': #forecast task
        data_task = generate_forecast_task(dataset)
    elif args.mode == 'onestock': #one stock prediction task
        data_task = generate_onestock_task(dataset)
    else: # allocation task
        data_task = generate_allocation_task(dataset)

    if args.thub != '': #save task data
        data_task.push_to_hub(args.thub)
        