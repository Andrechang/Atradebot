# create dataset for training a financial model on Yahoo Finance data API, Google News
# collect news data for stocks during valley and peak days for a stock

# prompt creation: news + stock_data + date -> prompt 

import os
from argparse import ArgumentParser
import datasets
from datasets import Dataset
from atradebot.data.yf_data import get_yf_data_some, get_yf_detail
from atradebot.utils.utils import find_peaks_valleys, filter_points
from atradebot.params import DATE_FORMAT
from atradebot.utils.utils import find_business_day
from datetime import datetime
import pandas as pd
from atradebot.data.get_news import get_news


LLM_INSTRUCTION = """You are a seasoned stock market analyst. \
    You are given past news and basic financials from companies. \
    Your task is to provide prediction for the companies' stock price movement and forecast possible future news. 
    Your answer format should be as follows:\n\n {'Price Change Future Prediction': price percent_change (%), 'Future news': text}\n"""
    
def format_chat_template(tokenizer, row): 
    """
    Format a chat template using a given row of data.

    Args:
        tokenizer: An object that handles text tokenization.
        row (dict): A dictionary containing 'instruction' and 'response' keys.

    Returns:
        dict: The updated row with 'text' key containing the formatted chat template.
    """    
    row_json = [{"role": "system", "content": LLM_INSTRUCTION },
                {"role": "user", "content": row["instruction"]},]
    if 'response' in row:    
        row_json += [{"role": "assistant", "content": row["response"]}]
    
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

def construct_prompt(stock:str, 
                     present_date:datetime.date, 
                     past_date:datetime.date, 
                     futuredays:int=5, 
                     newsapi:str='dataset',
                     num_results = 5):
    """
    Construct a prompt for a stock based on historical news data.

    Args:
        stock (str): The stock symbol for which the prompt is being constructed.
        present_date (datetime.date): The present date.
        past_date (datetime.date): The past date to retrieve news from.
        futuredays (int, optional): Number of future days to consider for the prompt. Default is 5.
        newsapi (str, optional): Source of news data. Default is 'dataset'.

    Returns:
        tuple: A tuple containing the constructed prompt and a DataFrame of news data.

    Raises:
        None
    """    
    timeperiod = [past_date.strftime(DATE_FORMAT), present_date.strftime(DATE_FORMAT)]
    news = get_news(stock, timeperiod, num_results=num_results, news_source=newsapi)
    if len(news) == 0: # if cant find news, use random past dates from dataset news
        news = get_news(stock, [], num_results=num_results, news_source='dataset')

    news = pd.DataFrame(news)

    # prune to keep prompt short (avoid OOM)
    # if len(news) > num_results: # truncate to 5 news
    #     news = news.sample(num_results)
    # news = news.drop_duplicates(subset=['Date'])# remove repeated date news

    now = True if present_date == datetime.today().date() else False
    prompt = build_context(news, stock, futuredays, now=now)
    # print('Prompt context len: ', len(prompt), len(news))
    return prompt, news

def collect_events(stock:str, past_date:str, present_date:str, ret=False):
    """
    Collect events to gather news.

    Args:
        stock (str): Stock id.
        past_date (str): Date start to collect data in format yyyy-mm-dd (past).
        present_date (str): Date end to collect data in format yyyy-mm-dd (today).
        ret (bool, optional): If True, return random events, defaults to False.

    Returns:
        list[pandas.Timestamp]: List of dates to collect news.
    """    
    
    data = get_yf_data_some(stock, past_date=past_date, present_date=present_date)
    if data.empty or ret: # generate random events if no data
        l = data.index.to_list()
        events = [ll for i, ll in enumerate(l) if i%2]
        return events

    data_mean = []
    data_stock = data['Close'].values.squeeze()
    for i in range(len(data_stock)-3):
        data_mean.append(data_stock[i:i+3].mean())

    # collect when to gather news
    p, v = find_peaks_valleys(data_mean)
    # index of data_mean that are peaks and valleys
    peak_idx, valley_idx = filter_points(data_mean, p, v) 
    events = peak_idx + valley_idx #concat lists
    events += [2] # add second day as first event to collect news
    events.sort()
    events_dates = [data.index[event] for event in events]
    return events_dates, data


def build_context(news, stock, futuredays, now=False):
    """
    Build a prompt with news about a specific stock and request for the price percentage change after a certain number of days.

    Args:
        news (DataFrame): A DataFrame containing news data with columns 'Date' and 'News'.
        stock (str): The name of the stock.
        futuredays (int): The number of days in the future to predict the stock price change.
        now (bool, optional): If True, include today's information. Defaults to False.

    Returns:
        str: A prompt with news about the stock and a request for the price percentage change after a certain number of days.
    """   

    # Add stock description and company info
    company, company_text, company_person, company_info = get_yf_detail(stock)
    prompt = f"{stock}: {company_text} \n"
    if now: # can add today's info if not using for backtesting
        prompt += f"{company_person} \n Company Details: {company_info} \n"
    
    # Add News
    if len(news) > 0:
        prompt += f"Here is a list of news about {stock}: \n"
        
        news_grouped = news.groupby('Date')['News'].apply(' '.join).reset_index() # concatenate news for same date
        for _, row in news_grouped.iterrows():
            prompt += f"Date: {row['Date']}, News: {row['News']} \n"

    # Add instruction
    prompt += f"Please, provide the price percentage change (%) for the {stock} stock price after {futuredays} days in the future: \n"
    return prompt


def create_prompt(dataset:Dataset, stock, backdays, futuredays, getevent=False):
    # for llama3
    """
    Create a prompt dataset for a given stock.

    Args:
        dataset (Dataset): The dataset containing stock data.
        stock (str): The stock ticker symbol to create prompt for.
        backdays (int): Number of days to look back.
        futuredays (int): Number of days to look into the future.
        getevent (bool, optional): Flag to determine if events should be collected. Defaults to False.

    Returns:
        Dataset: A dataset containing prompts and responses for the stock.

    Raises:
        None
    """    
    dhf = dataset.to_pandas()
    # filter for a stock
    dhf = dhf[dhf['Tickers'] == stock]

    # get max and min dates in dataset 
    max_date = max(dhf['Date'])
    min_date = min(dhf['Date'])
    # gather interesting dates
    if getevent:
        events, data = collect_events(stock, min_date, max_date)
    else:
        events = None
        data = get_yf_data_some(stock, min_date, max_date)
    data = data.reset_index() # reset index to get date

    if not events:
        # get random dates between min and max date
        events = data['Date'].sample(int(len(data['Date']) * 0.7))

    # collect group of news and financials for that stock up to some dates behind (backdays) the interesting date
    llm_dataset = [] # TODO need streaming 
    for event in events:
        past = find_business_day(event, -backdays)
        if past < min_date:
            continue
        future = find_business_day(event, futuredays)
        if future > max_date:
            continue
        # get news and financials for that stock up to backdays behind the interesting date
        # news = dhf.filter(lambda x: x['Date'] <= event and x['Date'] >= past)
        news = dhf[(dhf['Date'] <= event) & (dhf['Date'] >= past)]
        if len(news) == 0:
            continue
        # TODO: add some financials
        # create prompt
        prompt = build_context(news, stock, futuredays)
        # create response
        # news_future = dhf.filter(lambda x: x['Date'] == future)
        news_future = dhf[dhf['Date'] == future]
        if len(news_future) == 0:
            continue
        resp = ""
        news_grouped = news_future.groupby('Date')['News'].apply(' '.join).reset_index()
        for _, row in news_grouped.iterrows():
            resp += f"Date: {row['Date']}, News: {row['News']} \n"
  
        fval = data.loc[data['Date'] == future, 'Close']
        if len(fval) == 0:
            continue
        
        pred = fval.values[0] - data.loc[data['Date'] == event, 'Close'].values[0]
        pred = pred / data.loc[data['Date'] == event, 'Close'].values[0] * 100
        response = {"Prediction": pred[0], "Future news": resp}
        llm_dataset.append({'instruction': prompt, 'response': response})

    datas = Dataset.from_list(llm_dataset)
    return datas

if __name__ == "__main__":
    pass
