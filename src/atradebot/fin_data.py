# create dataset for training a financial model on Yahoo Finance data API, Google News

import os
from argparse import ArgumentParser
import pandas as pd
import yfinance as yf
import numpy as np
from atradebot import main, news_utils
from datasets import Dataset
import time
import json
from datasets import load_dataset, Dataset
from tqdm import tqdm


# find peaks and valleys
def find_peaks_valleys(array):
    peaks = []
    valleys = []
    off = 5
    for i in range(off, len(array) - off, off):
        if array[i - off] < array[i] > array[i + off]:
            peaks.append(i)
        elif array[i - off] > array[i] < array[i + off]:
            valleys.append(i)
    return peaks, valleys


# ignore peaks and valleys that are too close to each other
def filter_points(data, peak_idx, valley_idx):
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


def collect_events(stock, start_date, end_date, ret=False):
    """collect events to gather news
    Args:
        stock (str): stock id
        start_date (str): date start to collect data in format yyyy-mm-dd
        end_date (str): date end to collect data in format yyyy-mm-dd
        ret (bool, optional): if True, return random events. Defaults to False.
    Returns:
        list[pandas.Timestamp]: list of dates to collect news
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
    p, v = find_peaks_valleys(data_mean)
    # index of data_mean that are peaks and valleys
    peak_idx, valley_idx = filter_points(data_mean, p, v) 
    events = peak_idx + valley_idx #concat lists
    events += [2] # add second day as first event to collect news
    events.sort()
    events_dates = [data.index[event] for event in events]
    return events_dates


def gen_news_dataset(stocks, start_date, end_date, num_news=5, sample_mode = 'sp500', news_source = 'finhub'):
    """collect news at specific dates for stocks
    GoogleSearch API only allows 100 requests per day
    Args:
        stocks (list[str]): list of stock ids
        start_date (str): date start to collect data in format yyyy-mm-dd
        end_date (str): date end to collect data in format yyyy-mm-dd
        to_hub (bool, optional): save to huggingface hub. Defaults to False.
        num_news (int, optional): number of news to collect. Defaults to 5.
        sample_mode (str, optional): 'sp500', sample or 'stocks'. Defaults to 'sp500'.
            sample: collect news in random dates
            stocks: collect news based on up and down changes for each stock
            sp500: collect news based on up and down changes for SP500
        news_source (str, optional): 'google' or 'finhub'. Defaults to 'finhub'.
    Returns:
        huggingface dataset: dataset in hugginface format
    """    
    assert sample_mode in ['sp500', 'stocks', 'samples']
    assert news_source in ['google', 'finhub']

    all_news = []

    done = [] #keep track of stocks already done
    # to_save = [] #keep track of stocks that need to be saved
    # if os.path.exists("saved_stocks.json"):
    #     with open("saved_stocks.json", "r") as file:
    #         done = json.load(file)

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
            try:
                if news_source == 'google':
                    news, _, _ = news_util.get_google_news(stock=stock, num_results=num_news, time_period=[start, end])
                else:
                    news, _, _ = news_util.get_finhub_news(stock=stock, num_results=num_news, time_period=[start, end])

                if news == []:
                    print(f"Can't collect news for {stock} dates {start} to {end}")
            except:
                print(f"Can't collect news for {stock} dates {start} to {end}")
                continue
            all_news += news
        time.sleep(5)

        # done.append(stock)
        # to_save.append(stock)
        # if to_hub and len(to_save) > 5:
        #     dataset = Dataset.from_list(all_news)
        #     dataset.push_to_hub(f"achang/{HF_news}_{to_save[0]}_{to_save[-1]}")
        #     to_save = []
        #     time.sleep(30)
        # with open("saved_stocks.json", "w") as file:
        #     json.dump(done, file)

    dataset = Dataset.from_list(all_news)
    return dataset


def generate_forecast_task(data, to_hub=False): 
    #json dataset for lit-llama
    file_data = []
    for sample in data:
        forecast = main.get_forecast(sample['stock'], sample['date'])
        file_data.append({'instruction': f"what is the forecast for {sample['stock']}", 
                        'input':f"{sample['date']} {sample['title']} {sample['snippet']}", 
                        'output':f"{round(forecast[0], 2)}, {round(forecast[1],2)}, {round(forecast[2],2)}"})   
    dataset = Dataset.from_pandas(pd.DataFrame(data=file_data))
    with open("stock_forecast.json", "w") as outfile:
        json.dump(file_data, outfile)
    return dataset


def generate_allocation_task(data, to_hub=False): 

    # collect news for similar dates on some stocks


    file_data = []
    for sample in data:
        forecast = main.get_forecast(sample['stock'], sample['date'])

        #collect news for similar dates on some stocks

        #get forecast and apply simple buy/sell strategy

        #generate output based on allocation

        file_data.append({'instruction': f"what is the forecast for {sample['stock']}", 
                        'input':f"{sample['date']} {sample['title']} {sample['snippet']}", 
                        'output':f"{round(forecast[0], 2)}, {round(forecast[1],2)}, {round(forecast[2],2)}"})   

    dataset = Dataset.from_pandas(pd.DataFrame(data=file_data))
    with open("stock_alloc.json", "w") as outfile:
        json.dump(file_data, outfile)
    return dataset


def combine_datasets():
    #combine all news datasets into one
    #1) benzinga: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests?resource=download&select=raw_analyst_ratings.csv
    # https://github.com/miguelaenlle/Scraping-Tools-Benzinga/blob/master/scrape_benzinga.py
    #2) finviz: https://www.kaggle.com/datasets?search=finviz

    return None


def get_arg(raw_args=None):
    parser = ArgumentParser(description="parameters")
    parser.add_argument('-m', '--mode', type=str,
                        default='forecast', help='Create dataset for task modes: forecast | allocation ')
    parser.add_argument('-t', '--thub', type=str,
                        default='', help='push to hub folder name for task dataset')
    parser.add_argument('-r', '--rhub', type=str,
                        default='achang/stocks_grouped', help='push to hub folder name for raw news data')
    parser.add_argument('--start_date', type=str, default="2022-08-31", help='start date for trading analysis')
    parser.add_argument('--end_date', type=str, default="2023-06-20", help='end data for trading analysis')
    parser.add_argument('--stocks', type=str, default="AAPL AMZN MSFT NVDA TSLA", help='stocks to analize')
    args = parser.parse_args(raw_args)
    return args


if __name__ == "__main__":
    args = get_arg()
    stocks = args.stocks.split()
    if args.stocks == 'sp500':
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        stocks = sp500.Symbol.to_list()
    
    dataset = gen_news_dataset(stocks, args.start_date, args.end_date, 
                                sample_mode='sp500', news_source='finhub', num_news=20)
    if args.rhub != '':
        dataset.push_to_hub(args.rhub)

    if args.mode == 'forecast':
        data_task = generate_forecast_task(dataset)
    else:
        data_task = generate_allocation_task(dataset)

    if args.hub != '':
        data_task.push_to_hub(args.hub)
        