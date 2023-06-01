

''' TODO list:

user input: [x]
    balance (INIT_CASH)
    time horizon (TIMEFRAME)

chrono job:[x]
    set to run analysis every month
    set to update news every day

collect data:
    dynamic data: check every day
    - check news every day[x]
    - calculate sentiment for decision during specific dates[x]

    static data: check when need decision
    - historical data <<--
    - balance sheet

    decision: given table of stocks or areas to invest 
        - news sentiment[x]
        - historic data info
        - balance sheet
        - personel (linkedin)


decision algorithm: rank best to buy and sell
    - simple average of news sentiment
    - ask gpt
    - fundamental analysis
    - technical analysis

more stuff:
    - buy/hold [x]
    - sell
    - add/swap to new stocks
    - look into ETFs

output:
    - average-cost stat output profile files .csv [x]
    - email alert when to put money every month
    - graph of prediction
    - optional: alpaca api to auto-execute trade

'''

import torch
import pandas as pd
import tweepy
import re 
from transformers import AutoModelForImageClassification, AutoImageProcessor
import cv2
from PIL import Image, ImageDraw
from argparse import ArgumentParser
import os
from datetime import date, datetime
import yfinance as yf
import shutil 
import math

import json
import argparse
import requests
from bs4 import BeautifulSoup
import trafilatura
from nltk import tokenize
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from dateutil.relativedelta import relativedelta
from tqdm import tqdm   
import openai

# user finance plan input
# TODO: add saved income to balance
TIMEFRAME = 1*365 # days: 1yrs 
INIT_CASH = 10000 # investment 50k over a year
INTERVAL_ANALYSIS = 15 # days to analyze and invest 
INVEST_AMOUNT = INIT_CASH/(TIMEFRAME/INTERVAL_ANALYSIS) # amount to invest every INTERVAL_ANALYSIS days
STOCKS2CHECK = ['AAPL','ABBV','AMZN','ASML','BHP','COST','GOOGL','JNJ','KLAC','LLY','LRCX','MSFT','NVDA','TSLA'] # list of stocks to check

# profile data
DIR = 'mydata'
PATH_P = os.path.join(DIR, 'myportifolio.csv')# Name,Qnt,UCost (unit cost),BaseCost,Price (current price),Value (current Value), LongGain (Qnt), ShortGain (Qnt)
PATH_T = os.path.join(DIR, 'mytaxtime.csv')# Name, TB (time bought), Qnt
PATH_S = os.path.join(DIR, 'mysold.csv')# Name, TS (time sold), Qnt, Proceed
PATH_B = os.path.join(DIR, 'mybalance.csv')# Time,Cash,Stock,Total
PATH_N = os.path.join(DIR, 'news.csv')# Time,Name,Text,Score,Link
DATE_FORMAT = "%Y-%m-%d"


def get_arg(raw_args=None):
    parser = ArgumentParser(description="parameters")
    parser.add_argument('-m', '--mode', type=str,
                        default='run', help='Modes: init | run | debug ')
    args = parser.parse_args(raw_args)
    return args


def pd_append(df, dict_d):
    return pd.concat([df, pd.DataFrame.from_records([dict_d])])

#create a new profile
def new_profile():
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    dfp = pd.DataFrame(columns=['Name','Qnt','UCost','BaseCost','Price','Value','LongGain','ShortGain'])
    for stock in STOCKS2CHECK:
        dfp = pd_append(dfp, {'Name': stock, 'Qnt': 0, 'UCost': 0, 'BaseCost': 0, 'Price': 0, 'Value': 0, 'LongGain': 0, 'ShortGain': 0})
    dft = pd.DataFrame(columns=['Name','TB','Qnt'])
    dfs = pd.DataFrame(columns=['Name','TS','Qnt','Proceed'])
    dfb = pd.DataFrame(columns=['Time','Cash','Stock','Total'])
    dfb = pd_append(dfb, {'Time': date.today().strftime(DATE_FORMAT), 'Cash': INIT_CASH, 'Stock': 0, 'Total': INIT_CASH})
    news = pd.DataFrame(columns=['Time','Name','Text','Score','Link'])
    news.to_csv(PATH_N, index=False)
    dfp.to_csv(PATH_P, index=False)
    dft.to_csv(PATH_T, index=False)
    dfs.to_csv(PATH_S, index=False)
    dfb.to_csv(PATH_B, index=False)
    print("New profile created. Plan to invest {}$ over {} days. Each interval of {} days will invest {}$ ".format(INIT_CASH, 
        TIMEFRAME, INTERVAL_ANALYSIS, INVEST_AMOUNT))

#save backup
def save_back():
    dir_bk = DIR + str(datetime.today().strftime(DATE_FORMAT))
    if not os.path.exists(dir_bk):
        os.mkdir(dir_bk)
    shutil.copy(PATH_P, dir_bk)         
    shutil.copy(PATH_T, dir_bk)  
    shutil.copy(PATH_B, dir_bk)  
    shutil.copy(PATH_S, dir_bk)  
    shutil.copy(PATH_N, dir_bk) 
    print("Backup saved at {}".format(dir_bk))

def get_price(stock):
    ticker = yf.Ticker(stock).info
    return ticker['regularMarketOpen']

class TradingBot:
    def __init__(self, args):
        self.dfp = pd.read_csv(PATH_P) # portifolio
        self.dfp.fillna(0, inplace=True)
        self.dft = pd.read_csv(PATH_T) # tax time
        self.dfb = pd.read_csv(PATH_B) # balance
        self.dfs = pd.read_csv(PATH_S) # sold
        
        self.news = pd.DataFrame(columns=['Time','Name','Text','Score','Link'])#news for stocks
        self.stats = {} #stats for stocks: dict {stock: history data}

        self.stock_rank = {} #ranked stocks: dict{stock: {decision: buy/sell,  news: [ranked news], info: "gpt analysis" } }

        self.general_news = {} # TODO: general info and general finance data (monitor Indexes): dict{bull/bear:[news]}
        # self.reload_news()
        # self.init_tweet()
        self.init_model()


    # Authenticate to Twitter
    def init_tweet(self):
        auth = tweepy.OAuthHandler("UkIxHV1myPKpxf2bEwd1WCmM1", "ifh3rDZDHG8C4tu2JsgtgDbQGD77WgkdgL5t1P7zyHp3c9Dero")
        auth.set_access_token("177934931-wioPKK09BQF5jNLKyUVSvH4GlJRh0LCZqzorXHk5", "QnmDsrBkdbSTxpi7LFz4MtdMB83Rgt8B7vmIF5KsuES0Z")
        # Create API object
        self.api = tweepy.API(auth)
        try:
            self.api.verify_credentials()
            print("Authentication ok")
        except:
            print("Error in authentication")
    
    # get NLP model to analyze sentiment
    def init_model(self):
        finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
        tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.sentiment_analyzer = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

    # collect google search text and sentiment score
    def get_google_news(self, query, num_results=10, max_length=512):
        headers = {
                "User-Agent":
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36"
            }

        search_req = "https://www.google.com/search?q="+query+"&gl=us&tbm=nws&num="+str(num_results)+""

        news_results = []
        
        # get webpage content
        response = requests.get(search_req, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        for el in soup.select("div.SoaBEf"):
            sublink = el.find("a")["href"]
            downloaded = trafilatura.fetch_url(sublink)
            html_text = trafilatura.extract(downloaded)
            if html_text:
                sentences = tokenize.sent_tokenize(html_text)
                # truncate sentences that are too long
                for i, s in enumerate(sentences):
                    if len(s) > max_length:
                        sentences[i] = sentences[i][:max_length]

                sentiment = self.sentiment_analyzer(sentences)
                sum = 0
                neutrals = 0
                if len(sentiment) > 0:
                    for r in sentiment: 
                        sum += (r["label"] == "Positive")
                        neutrals += (r["label"] == "Neutral")

                    den = len(sentiment)-neutrals
                    sentiment = sum/den if den > 0 else 1.0 # as all neutral

                    news_results.append(
                        {
                            "link": el.find("a")["href"],
                            "title": el.select_one("div.MBeuO").get_text(),
                            "snippet": el.select_one(".GI74Re").get_text(),
                            "date": el.select_one(".LfVVr").get_text(),
                            "source": el.select_one(".NUnG9d span").get_text(),
                            "text": html_text,
                            "sentiment": sentiment,
                        }
                    )

        return news_results, search_req

    # get the news from: google, TODO yfinance, twitter, 
    # save news into PATH_N using pandas, set: self.news
    def get_news(self):
        for stock in tqdm(list(self.dfp['Name'])):
            #twitter
            # posts = self.api.user_timeline(screen_name="BillGates", count = 100, lang ="en", tweet_mode="extended")
            # df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])

            # get google text news
            gnews, _ = self.get_google_news(stock)
            # Time,Name,Text,Score,Link
            for gnew in gnews:
                self.news = pd_append(self.news, {'Time': gnew["date"], 'Name': stock, 
                    'Text': gnew["text"], 'Score': gnew["sentiment"], 'Link': gnew["link"], 'Snippet': gnew["snippet"]})

        if os.path.isfile(PATH_N): #concat with old news
            old_news = pd.read_csv(PATH_N)
            self.news = pd.concat([old_news, self.news])        

        #filter repeated news
        self.news = self.news.drop_duplicates(subset=['Text'])

        self.news.to_csv(PATH_N, index=False) #save news
        print("News saved in: ", PATH_N)

    # get the historical data
    # set: self.stats
    def get_stats(self):
        end_date = date.today().strftime(DATE_FORMAT)
        past_year = date.today() - relativedelta(years=2)
        start_date = past_year.strftime(DATE_FORMAT) # Example start date in yyyy-mm-dd format
        for stock in list(self.dfp['Name']):
            #get historical data
            df_data = yf.download(stock, start=start_date, end=end_date)
            self.stats[stock] = df_data
        

    # rank stocks based on data
    # return: dict{stock: {decision: buy/sell, info: "gpt analysis" } }
    def get_rank(self):
        self.stock_rank = {}
        # self.get_stats()
        # TODO: +historical data analysis

        # TODO: AutoGPT? FinNLP analysis
        

        # simple mean of sentiment score ranking
        mean_score = []
        for stock in list(self.dfp['Name']):
            filtered_df = self.news[self.news['Name'] == stock]
            # filtered_df.sort_values(by='Score', ascending=False)
            mean = filtered_df['Score'].mean()
            mean_score.append((stock,mean))
        mean_score.sort(key=lambda x: x[1], reverse=True)
        # distribute money based on ranking
        spend = INVEST_AMOUNT
        distrib = [0.6, 0.3, 0.2, 0.1]
        idx = 0
        while spend > 0:
            price = get_price(mean_score[idx][0])
            s = INVEST_AMOUNT*distrib[idx]
            s = math.ceil(s/price)
            spend -= s*price
            self.stock_rank[mean_score[idx][0]] = {'decision': s}


    #execute buy/sell suggestions from get_rank
    def execute(self, dict_rank):

        for stock, value in dict_rank.values():
            assert stock in list(self.dfp['Name']), "Stock not in portifolio"
            idx = self.dfp.loc[self.dfp['Name'] == stock].index[0]
            ticker = yf.Ticker(stock).info
            price = self.dfp.at[idx, 'Price'] = ticker['regularMarketOpen']
            if value['decision'] > 0: #buy
                cost = price*value['decision']
                if cost > self.dfb.at[0, 'Cash']: # check balance
                    print("Not enough money")
                    break
                else:
                    self.dft = pd_append(self.dft, {'Name': stock, 'TB': date.today().strftime(DATE_FORMAT), 'Qnt': value['decision']})
                    self.dfp.at[idx, 'BaseCost'] += cost
                    self.dfb.at[0, 'Cash'] -= cost
            elif value['decision'] < 0: #sell
                proceed = price*value['decision']
                #get oldest qnt and reduce it
                self.dfs = pd_append(self.dfs, {'Name': stock, 'TS': date.today().strftime(DATE_FORMAT), 
                    'Qnt': value['decision'], 'Proceed': proceed})
                self.dfp.at[idx, 'BaseCost'] -= proceed
                self.dfb.at[0, 'Cash'] += proceed
                
        # update portifolio
        self.update_all()
        return



    #update calculated numbers in files after execute 
    def update_all(self):
        # check portifolio numbers
        for index, row in self.dfp.iterrows():
            name = row['Name']
            self.dfp.at[index, 'Price'] = get_price(name)
            self.dfp.at[index, 'UCost'] = self.dfp.at[index, 'BaseCost']/self.dfp.at[index, 'Qnt'] if self.dfp.at[index, 'Qnt'] > 0 else 0 # UCost=BaseCost/Qnt
            self.dfp.at[index, 'Value'] = self.dfp.at[index, 'Price']*self.dfp.at[index, 'Qnt']# Value=Price*Qnt
            # check taxtime Qnt
            # ShortGain=Qnt-BaseCost
            # LongGain=Qnt-BaseCost
            dtoday = datetime.today()
            dft_index = self.dft.loc[self.dft['Name'] == name]
            LongGain, ShortGain = 0, 0
            for _, rowt in dft_index.iterrows():
                d1 = datetime.strptime(rowt['TB'], DATE_FORMAT)
                delta = dtoday - d1
                if delta.days > 365:
                    LongGain += rowt['Qnt'] 
                else:
                    ShortGain += rowt['Qnt']
            self.dfp.at[index, 'LongGain'] = LongGain
            self.dfp.at[index, 'ShortGain'] = ShortGain

        # check balance
        total_stock = self.dfp['Value'].sum()
        self.dfb.at[0, 'Stock'] = total_stock
        self.dfb.at[0, 'Time'] = str(datetime.now().strftime(DATE_FORMAT))

        #update files
        self.dfp.to_csv(PATH_P, index=False)
        self.dft.to_csv(PATH_T, index=False)
        self.dfb.to_csv(PATH_B, index=False)
        self.dfb.to_csv(PATH_S, index=False) 
        print("Updated files at {}".format(DIR))
 

    #TODO: search new stock to add
    def add_stock(self):
        pass
    #TODO: suggest drop a stock
    def remove_stock(self):
        pass

if __name__ == "__main__":
    args = get_arg()
    if args.mode == 'init':
        new_profile()
    elif args.mode == 'run' or args.mode == 'debug':    
        bot = TradingBot(args)# Create an instance of the trading bot
        today_date = datetime.today()
        interval_date = today_date - datetime.strptime(bot.dfb['Time'].values[-1], DATE_FORMAT) 
        save_back() #create backup
        bot.get_news() #run news checks
        # print(bot.news)
        if (interval_date.days >= INTERVAL_ANALYSIS or args.mode == 'debug'):
            bot.get_rank() # rank for suggestion
            print(bot.stock_rank)
            approve = input("Do you want to execute? (y/n)")
            if approve == 'y':
                bot.execute()
            else:
                pass

