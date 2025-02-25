# collect news via api every day. Use with cron job. 
# crontab -e
# 30 2 * * * collect.py
from atradebot.data.get_news import get_news
import random
from datetime import datetime, timedelta, date
from atradebot.params import DATE_FORMAT, HF_LOCALDATA
import datasets
import os
from atradebot.utils.search import stock_research
import json
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
