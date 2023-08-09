# add different source to gather news and data here
# function input: stock, date, num_results
# function output: list of dict {"link", "title", "snippet", "date", "source", "text", "stock"}

import pandas as pd
# import tweepy
from argparse import ArgumentParser
import os
from datetime import date, datetime, timedelta
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import trafilatura
from trafilatura.settings import use_config
import finnhub

trafilatura_config = use_config()
trafilatura_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
DATE_FORMAT = "%Y-%m-%d"
finnhub_client = finnhub.Client(api_key="cic7bdhr01ql0uqkr15gcic7bdhr01ql0uqkr160")

def get_google_news(stock, num_results=10, time_period=[]):
    """collect google search text 
    limit to 100 results per day

    :param stock: list of stock ids
    :type stock: List[str]
    :param num_results: number of news to collect, defaults to 10
    :type num_results: int, optional
    :param time_period: time_period=['2019-06-28' (start_time), '2019-06-29' (end_time)], defaults to []
    :type time_period: list, optional
    :return: list of news dict {"link", "title", "snippet", "date", "source", "text", "stock"}, search query, soup html searched
    :rtype: list of dict, str, html
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
            news_results.append(
                {
                    "link": el.find("a")["href"],
                    "title": el.select_one("div.MBeuO").get_text(),
                    "snippet": el.select_one(".GI74Re").get_text(),
                    "date": el.select_one(".LfVVr").get_text(),
                    "source": el.select_one(".NUnG9d span").get_text(),
                    "text": html_text,
                    "stock": stock,
                }
            )
    return news_results, search_req, soup

def get_finhub_news(stock, num_results=10, time_period=[]):
    """collect from finhub news: https://finnhub.io/
    only up to 1 year old news and 60 requests per minute

    :param stock: list of stock ids
    :type stock: List[str]
    :param num_results: number of news to collect, defaults to 10
    :type num_results: int, optional
    :param time_period: time_period=['2019-06-28' (start_time), '2019-06-29' (end_time)], defaults to []
    :type time_period: list, optional
    :return: list of news in a dict{"link", "title", "snippet", "date", "source", "text", "stock"}
    :rtype: list of dict, None, None
    """    
    results = finnhub_client.company_news(stock, _from=time_period[0], to=time_period[1])
    news_results = []
    for result in results:
        try:
            downloaded = trafilatura.fetch_url(result['url'])
        except:
            continue
        html_text = trafilatura.extract(downloaded, config=trafilatura_config)
        if html_text:
            news_results.append(
                {
                    "link": result['url'],
                    "title": result['headline'],
                    "snippet": result['summary'],
                    "date": datetime.fromtimestamp(result['datetime']),
                    "source": result['source'],
                    "text": html_text,
                    "stock": stock,
                }
            )
    search_req, soup = None, None
    return news_results, search_req, soup