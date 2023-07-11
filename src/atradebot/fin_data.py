# create dataset for traininig sentiment model

import os
import pandas as pd
import yfinance as yf
import numpy as np
from atradebot import main
from datasets import Dataset
import time
import json
from datasets import load_dataset, Dataset
from tqdm import tqdm

HF_forecast = "stock_forecast_sp500_2010q1_2023q2"
HF_news = "stock_news_sp500_2010q1_2023q2"


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


# collect news around stock price peaks and valleys
def gen_news_dataset(stocks, start_date, end_date, to_hub=False, num_news=10):
    # data: pandas dataframe
    # to_hub: bool, if true, upload to huggingface hub

    all_news = []
    done = [] #keep track of stocks already done
    to_save = [] #keep track of stocks that need to be saved
    if os.path.exists("saved_stocks.json"):
        with open("saved_stocks.json", "r") as file:
            done = json.load(file)

    for stock in tqdm(stocks):
        if stock in done:
            continue
        data = yf.download(stock, start=start_date, end=end_date)
        if data.empty:
            continue

        data_mean = []
        for i in range(len(data['Close'])-10):
            data_mean.append(data['Close'][i:i+10].mean())

        p, v = find_peaks_valleys(data_mean)
        # index of data_mean that are peaks and valleys
        peak_idx, valley_idx = filter_points(data_mean, p, v) 
        events = peak_idx + valley_idx #concat lists
        events += [2] # add second day as first event to collect news
        events.sort()
        print(f'{stock}, events {len(events)}')
        for event in events:
            start = main.business_days(data.index[event], -1)#one day before
            start = start.strftime(main.DATE_FORMAT)
            end = main.business_days(data.index[event], +1)#one day after
            end = end.strftime(main.DATE_FORMAT)
            try:
                news, _, _ = main.get_google_news(stock=stock, num_results=num_news, time_period=[start, end])
                if news == []:
                    print(f"Can't collect news for {stock} dates {start} to {end}, google maxed out")
            except:
                print(f"Can't collect news for {stock} dates {start} to {end}")
                continue
            all_news += news
            time.sleep(10)

        done.append(stock)
        to_save.append(stock)

        dataset = Dataset.from_list(all_news)

        if to_hub and len(to_save) > 5:
            dataset.push_to_hub(f"achang/{HF_news}_{to_save[0]}_{to_save[-1]}")
            to_save = []
            time.sleep(30)

        with open("saved_stocks.json", "w") as file:
            json.dump(done, file)


    return dataset


def generate_json(data, to_hub=False): 
    #json dataset for lit-llama
    file_data = []
    for sample in data:
        forecast = main.get_forecast(sample['stock'], sample['date'])
        file_data.append({'instruction': f"what is the forecast for {sample['stock']}", 
                        'input':f"{sample['date']} {sample['title']} {sample['snippet']}", 
                        'output':f"{round(forecast[0], 2)}, {round(forecast[1],2)}, {round(forecast[2],2)}"})   
    dataset = Dataset.from_pandas(pd.DataFrame(data=file_data))
    if to_hub:
        dataset.push_to_hub(f"achang/{HF_forecast}")

    with open(f"{HF_forecast}.json", "w") as outfile:
        json.dump(file_data, outfile)


def combine_datasets():
    #combine all news datasets into one
    #1) benzinga: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests?resource=download&select=raw_analyst_ratings.csv
    # https://github.com/miguelaenlle/Scraping-Tools-Benzinga/blob/master/scrape_benzinga.py
    #2) finviz: https://www.kaggle.com/datasets?search=finviz

    return None


if __name__ == "__main__":
    
    # sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    # stocks = sp500.Symbol.to_list()
    stocks = ['AAPL','ABBV','AMZN','MSFT','NVDA','TSLA']

    start_date = "2018-01-31"  
    end_date = "2023-06-20"
    dataset = gen_news_dataset(stocks, start_date, end_date, to_hub=True)
    generate_json(dataset, to_hub=True)
