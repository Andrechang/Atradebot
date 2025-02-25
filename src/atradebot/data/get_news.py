# add different source to gather news and data here
# function input: stock, date, num_results
# function output: list of dict {"link", "title", "snippet", "date", "source", "text", "stock"}

import pandas as pd
# import tweepy
from argparse import ArgumentParser
import os
from datetime import date, datetime, timedelta
import requests
from bs4 import BeautifulSoup
import trafilatura
from trafilatura.settings import use_config
import finnhub
from tenacity import retry, stop_after_attempt, wait_fixed
from atradebot.data.__init__ import register_api
import traceback
from datasets import load_dataset
from gpt_researcher import GPTResearcher
import asyncio
from atradebot.params import NEWSCACHE
from atradebot.params import DATE_FORMAT, FHUB_API, HF_DATASET


LOADED_DATASET = load_dataset(HF_DATASET, split='train')
# LOADED_DATASET = load_from_disk(HF_DATASET)
LOADED_DATASETPD = LOADED_DATASET.to_pandas()

trafilatura_config = use_config()
trafilatura_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
finnhub_client = finnhub.Client(api_key=FHUB_API)


# create research news document for stocks
async def run_gptresearch(prompt):
    # Report Type
    """
    Run a GPTResearch operation using a specified prompt.

    Args:
        prompt (str): The prompt to use for the GPTResearch operation.

    Returns:
        str: The generated research report.

    Raises:
        None
    """    
    report_type = "research_report"

    # Initialize the researcher
    researcher = GPTResearcher(query=prompt, report_type=report_type, config_path=None)
    # Conduct research on the given query
    await researcher.conduct_research()
    # Write the report
    report = await researcher.write_report()
    
    return report


@register_api('google')
@retry(
    stop=stop_after_attempt(2),   # Retry up to 2 times
    wait=wait_fixed(60)  # Wait 60 seconds between retries
)
@NEWSCACHE.memoize(expire=604800)
def get_google_news(stock, num_results=10, time_period=[]):
    """
    Collect google search text and limit to 100 results per day.

    Args:
        stock (List[str]): List of stock ids
        num_results (int, optional): Number of news to collect, defaults to 10
        time_period (list, optional): Time period=['2019-06-28' (start_time), '2019-06-29' (end_time)], defaults to []

    Returns:
        list of news dict {"link", "title", "snippet", "date", "source", "text", "stock"}, search query, soup html searched
        str, html
    """     
    query = stock
    headers = {
            "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36"
        }
    if time_period:
        query += f"+before%3A{time_period[1]}+after%3A{time_period[0]}" # add time range     
    search_req = "https://www.google.com/search?q="+query+"&gl=us&tbm=nws&num="+str(num_results)+""
    #https://developers.google.com/custom-search/docs/xml_results#WebSearch_Request_Format
    news_results = []
    # get webpage content
    response = requests.get(search_req, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    for el in soup.select("div.SoaBEf"):
        sublink = el.find("a")["href"]
        downloaded = trafilatura.fetch_url(sublink)
        html_text = trafilatura.extract(downloaded, config=trafilatura_config)
        if html_text:
            date_ = el.select_one(".LfVVr").get_text()
            if 'ago' in date_:
                continue 
            date_i = datetime.strptime(date_, "%b %d, %Y")
            date_i = date_i.strftime(DATE_FORMAT)
            news_results.append(
                {
                    "link": el.find("a")["href"],
                    "title": el.select_one("div.MBeuO").get_text(),
                    "snippet": el.select_one(".GI74Re").get_text(),
                    "Date": date_i,
                    "source": el.select_one(".NUnG9d span").get_text(),
                    "News": html_text,
                    "Tickers": stock,
                }
            )
    return news_results, search_req, soup

@register_api('finhub')
@retry(stop=stop_after_attempt(2), wait=wait_fixed(60))
@NEWSCACHE.memoize(expire=604800)
def get_finhub_news(stock, num_results=10, time_period=[]):
    """
    collect from finhub news: https://finnhub.io/
    only up to 1 year old news and 60 requests per minute

    Args:
        stock (List[str]): list of stock ids
        num_results (int, optional): number of news to collect, defaults to 10
        time_period (list, optional): time_period=['2019-06-28' (start_time), '2019-06-29' (end_time)], defaults to []

    Returns:
        list of dict, None, None: list of news in a dict{"link", "title", "snippet", "date", "source", "text", "stock"}

    Raises:
        None
    """    
    if len(time_period) == 0:
        time_period = [str(date.today() - timedelta(days=10)), str(date.today())]

    results = finnhub_client.company_news(stock, _from=time_period[0], to=time_period[1])
    news_results = []
    for result in results:
        # try:
        #     downloaded = trafilatura.fetch_url(result['url'])
        # except:
        #     continue
        # html_text = trafilatura.extract(downloaded, config=trafilatura_config)
        html_text = result['summary']
        news_results.append(
            {
                "link": result['url'],
                "title": result['headline'],
                "snippet": result['summary'],
                "Date": datetime.fromtimestamp(result['datetime']).strftime(DATE_FORMAT),
                "source": result['source'],
                "News": html_text,
                "Tickers": stock,
            }
        )
    search_req, soup = None, None
    return news_results, search_req, soup


@register_api('dataset')
def get_dataset_news(stock, num_results=10, time_period=[]):
    """
    Get news dataset for a specific stock.

    Args:
        stock (str): The stock symbol for which to retrieve news data.
        num_results (int): The number of news results to return. Default is 10.
        time_period (list): A list containing a start and end date for filtering news data. Default is an empty list.

    Returns:
        tuple: A tuple containing news results as a list of dictionaries, search request, and soup.
    """    
    dataset = LOADED_DATASETPD
    dataset = dataset[dataset['Tickers'] == stock]
    if len(dataset) == 0:
        return [], None, None
    
    if len(time_period) == 0:
        dataset = dataset.sample(n=num_results)
    else:
        dataset = dataset[(dataset['Date'] >= time_period[0]) & (dataset['Date'] <= time_period[1])]

    news_results = dataset.to_dict(orient='records')
    search_req, soup = None, None
    return news_results, search_req, soup

@register_api('search')
@NEWSCACHE.memoize(expire=604800)  
def search_news(stock:str, num_results=10, time_period=[]):
    # get news for a stock (inference+dataset)
    """
    Search for news related to a specific stock within a time period.

    Args:
        stock (str): The stock symbol to search for news.
        num_results (int): Number of news results to retrieve. Default is 10.
        time_period (list): List containing two elements - the start and end date of the time period.

    Returns:
        list: A list of dictionaries containing information about the news articles found.

    Raises:
        None
    """    
    sysPrompt = 'You are a financial advisor. For the period between {} and {}, search the web for top {} \
        news that can alter the stock price of {}. ' 
    query = sysPrompt.format(num_results, time_period[0], time_period[1], stock)
    answer = asyncio.run(run_gptresearch(query))
    result = [{"link": '',
                    "title": '',
                    "snippet": '',
                    "Date": time_period[1],
                    "source": 'gptresearch',
                    "News": answer,
                    "Tickers": stock}]
    return result


def get_news(stock, time_period:list, num_results:int=10, news_source:str='google'):
    """
    Wrapper function to get news from different sources.

    Args:
        stock (List[str]): List of stock ids.
        time_period (list, optional): Time period=['2019-06-28' (start_time/past), '2019-06-29' (end_time/present)]. Defaults to [].
        num_results (int, optional): Number of news to collect. Defaults to 10.
        news_source (str, optional): News source ['google', 'finhub', 'newsapi', 'search']. Defaults to 'finhub'.

    Returns:
        list of news dict: {"link", "title", "snippet", "date", "source", "text", "stock"}, search query, soup html searched.
        str: Search query.
        html: Soup html searched.
    """    
    try:
        news = []
        if news_source == 'all':
            news, _, _ = get_google_news(stock=stock, num_results=num_results, time_period=time_period)
            news1, _, _ = get_finhub_news(stock=stock, num_results=num_results, time_period=time_period)
            news += news1
        elif news_source == 'google':
            news, _, _ = get_google_news(stock=stock, num_results=num_results, time_period=time_period)
        elif news_source == 'finhub':
            news, _, _ = get_finhub_news(stock=stock, num_results=num_results, time_period=time_period)
        elif news_source == 'dataset': #use news from dataset
            news, _, _ = get_dataset_news(stock=stock, num_results=num_results, time_period=time_period)
        else: #news_source == 'search': # generated from search.py
           news, _, _ = search_news(stock=stock, num_results=num_results, time_period=time_period)

        if news == []:
            print(f"Can't collect news for {stock}")
    except:
        traceback.print_exc()
        return []

    return news

if __name__ == "__main__":
    news = get_news(stock='AAPL', time_period=['2020-01-01', '2020-01-02'], num_results=3, news_source='google')
    print(news)
