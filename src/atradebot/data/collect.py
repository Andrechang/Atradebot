# collect news via api every day. Use with cron job. 
# crontab -e
# 30 2 * * * collect.py
from atradebot.data.get_news import get_news
import random
from datetime import datetime, timedelta, date
from atradebot.config import DATE_FORMAT, HF_LOCALDATA
import datasets
import os
import json
from atradebot.data.get_news import run_gptresearch
import asyncio
from atradebot.utils.utils import get_tickers

MARKET_CAP = 'large or medium cap'
SECTOR = 'information technology'

def stock_research(sell=False, skip_tickers:list=None):
    """
    Perform stock research and provide a list of the best 10 growth stocks in the information technology sector.
    Args:
        sell (bool): If True, search for stocks to sell. Default is False.
        skip_tickers (list): List of stock tickers to skip. Default is None.
    Returns:
        tuple: A tuple containing the list of tickers for the recommended stocks and the research answer.

    Raises:
        Any exceptions that may occur during the execution.
    """    
    cur_date = datetime.now().strftime(DATE_FORMAT)
    # get news to search for stocks
    if sell:
        sysPrompt = 'You are a financial advisor. Given today is {}, ' \
            'respond with a list of the worst 10 stocks in the {} sector.' \
            'List {} stocks that is expected to decline their stock price in a year in the future.'
        if skip_tickers is not None:
            skip_ = ', '.join(skip_tickers)
            sysPrompt += f' In your analysis, avoid the following stocks: {skip_}'
    else:
        sysPrompt = 'You are a financial advisor. Given today is {}, ' \
            'respond with a list of the best 10 growth stocks in the {} sector.' \
            'List {} stocks that is expected to grow their stock price in a year in the future.'
        
    query = sysPrompt.format(cur_date, SECTOR, MARKET_CAP)
    answer = asyncio.run(run_gptresearch(query))
    p = get_tickers(answer)
    return p, answer

if __name__ == "__main__": 
    # select some stocks to gather news
    stocks, answ = stock_research()
    today = date.today()
    past = today - timedelta(days=2)

    month = today.strftime('%m')
    year = today.strftime('%Y')
    HF_LOCALDATA += f'{month}_{year}.jsonl'
    print(f'{today} Collecting news for: {stocks}')
    news_text = [{"link": '',
                        "title": '',
                        "snippet": '',
                        "date": today.strftime(DATE_FORMAT),
                        "source": 'gptresearch',
                        "text": answ,
                        "stock": ' '.join(stocks)}]
    for stock in stocks:
        news = get_news(stock, 
                time_period=[past.strftime(DATE_FORMAT), today.strftime(DATE_FORMAT)], 
                num_results=5, 
                news_source='finhub')
        news_text.extend(news)

    print('Collected news:', len(news_text))
    with open(HF_LOCALDATA, "a") as file:  # Open in append mode
        for entry in news_text:
            file.write(json.dumps(entry) + "\n")

    print('All done!')
