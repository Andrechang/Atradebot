# main app code
# tries to keep track of your investments in a xlsx file, so that it will know what to do next time you run main again: sell or buy more


import pandas as pd
from argparse import ArgumentParser
import os
import math
import yfinance as yf
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import shutil

from atradebot.utils import is_business_day, business_days
from atradebot.utils import get_config, get_price, pd_append
from atradebot.utils import get_forecast
from atradebot import news_utils

def get_arg(raw_args=None):
    parser = ArgumentParser(description="parameters")
    parser.add_argument('-m', '--mode', type=str,
                        default='debug', help='Modes: run | debug ')
    parser.add_argument('-c', '--cfg', type=str,
                        default='default.yaml', help='Config file')
    args = parser.parse_args(raw_args)
    config = get_config(args.cfg)
    config.update(vars(args))
    return args, config


DATE_FORMAT = "%Y-%m-%d"

class TradingBot:
    def __init__(self, cfg):
        self.config = cfg
        # amount to invest every INTERVAL_ANALYSIS days
        self.invest_amount = self.config['INIT_CASH']/(self.config['TIMEFRAME']/self.config['INTERVAL_ANALYSIS']) 
        self.load_profile()

        self.stats = {} #stats for stocks: dict {stock: history data}
        self.stock_rank = {} #ranked stocks: dict{stock: {decision: buy/sell,  news: [ranked news], info: "analysis" } }
        
        # self.init_model() #init sentiment analysis model

    # PROFILE SAVING===================================================================================================
    
    def load_profile(self):
        if not os.path.isfile(self.config['SAVE_FILE']): #create a new profile
            self.holdings = pd.DataFrame(columns=['Name','Qnt','UCost','BaseCost','Price','Value','LongGain','ShortGain']) 
            for stock in self.config['STOCKS2CHECK']:
                self.holdings = pd_append(self.holdings, {'Name': stock, 'Qnt': 0, 'UCost': 0, 'BaseCost': 0, 
                    'Price': 0, 'Value': 0, 'LongGain': 0, 'ShortGain': 0})
            self.activity = pd.DataFrame(columns=['Name', 'Type','Time','Qnt','Value'])
            self.balance = pd.DataFrame(columns=['Time','Cash','Stock','Total'])
            self.balance = pd_append(self.balance, {'Time': date.today().strftime(DATE_FORMAT), 
                'Cash': self.config['INIT_CASH'], 'Stock': 0, 'Total': self.config['INIT_CASH']})
            self.news = pd.DataFrame(columns=['Time','Name','Text','Score','Link','Snippet'])
            with pd.ExcelWriter(self.config['SAVE_FILE']) as writer:
                self.holdings.to_excel(writer, float_format="%.2f", sheet_name='holdings', index=False)
                self.activity.to_excel(writer, float_format="%.2f", sheet_name='activity', index=False)
                self.balance.to_excel(writer, float_format="%.2f", sheet_name='balance', index=False)
                self.news.to_excel(writer, float_format="%.2f", sheet_name='news', index=False)
            print("New profile created. Plan to invest {}$ over {} days. Each interval of {} days will invest {}$ ".format(
                self.config['INIT_CASH'], 
                self.config['TIMEFRAME'], 
                self.config['INTERVAL_ANALYSIS'], 
                self.config['INVEST_AMOUNT']))
        else: #reload data from profile
            with open(self.config['SAVE_FILE'], 'rb') as reader:
                self.holdings = pd.read_excel(reader, sheet_name='holdings')
                self.holdings.fillna(0, inplace=True)
                self.activity = pd.read_excel(reader, sheet_name='activity') 
                self.balance = pd.read_excel(reader, sheet_name='balance')
                self.news = pd.read_excel(reader, sheet_name='news')

    #save backup
    def save_back(self):
        dir_bk = 'backup-' + str(datetime.today().strftime(DATE_FORMAT))
        if not os.path.exists(dir_bk):
            os.mkdir(dir_bk)
        shutil.copy(self.config['SAVE_FILE'], dir_bk)         
        print("Backup saved at {}".format(dir_bk))

        #update files
        with pd.ExcelWriter(self.config['SAVE_FILE']) as writer:
            self.holdings.to_excel(writer, float_format="%.2f", sheet_name='holdings', index=False)
            self.activity.to_excel(writer, float_format="%.2f", sheet_name='activity', index=False)
            self.balance.to_excel(writer, float_format="%.2f", sheet_name='balance', index=False)
            self.news.to_excel(writer, float_format="%.2f", sheet_name='news', index=False)
        print("Updated files at {}".format(self.config['SAVE_FILE']))


    # INFO GATHERING===================================================================================================
    # Authenticate to Twitter
    # def init_tweet(self):
    #     auth = tweepy.OAuthHandler("UkIxHV1myPKpxf2bEwd1WCmM1", "ifh3rDZDHG8C4tu2JsgtgDbQGD77WgkdgL5t1P7zyHp3c9Dero")
    #     auth.set_access_token("177934931-wioPKK09BQF5jNLKyUVSvH4GlJRh0LCZqzorXHk5", "QnmDsrBkdbSTxpi7LFz4MtdMB83Rgt8B7vmIF5KsuES0Z")
    #     # Create API object
    #     self.api = tweepy.API(auth)
    #     try:
    #         self.api.verify_credentials()
    #         print("Authentication ok")
    #     except:
    #         print("Error in authentication")
    
    # get the news from: google, TODO yfinance, twitter, 
    def get_news(self):
        print("Getting news:")
        for stock in self.config['STOCKS2CHECK']:
            print('\t', stock)
            #twitter
            # posts = self.api.user_timeline(screen_name="BillGates", count = 100, lang ="en", tweet_mode="extended")
            # df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
            #reddit
            #blind
            # get google text news
            gnews, _, _ = news_utils.get_google_news(stock)
            
            df = pd.DataFrame(gnews)
            self.news = pd.concat([self.news, df],ignore_index=True)

        #filter repeated news
        self.news = self.news.drop_duplicates(subset=['Text'])

    # get the historical data
    # set: self.stats
    def get_stats(self):
        end_date = date.today().strftime(DATE_FORMAT)
        past_year = date.today() - relativedelta(years=2)
        start_date = past_year.strftime(DATE_FORMAT) # Example start date in yyyy-mm-dd format
        for stock in list(self.holdings['Name']):
            #get historical data
            df_data = yf.download(stock, start=start_date, end=end_date)
            self.stats[stock] = df_data
        
    # STRATEGY===================================================================================================
    # rank stocks based on data
    # return: dict{stock: {decision: buy/sell, info: "gpt analysis" } }
    def get_rank(self):
        self.stock_rank = {}
        # self.get_stats()
        # TODO: +historical data analysis
        # TODO: AutoGPT? FinNLP analysis

        # simple mean of sentiment score ranking
        mean_score = []
        for stock in list(self.holdings['Name']):
            filtered_df = self.news[self.news['Name'] == stock]
            mean = filtered_df['Score'].mean()
            mean_score.append((stock, mean))
        mean_score.sort(key=lambda x: x[1], reverse=True)
        # distribute money based on ranking
        spend = self.invest_amount
        distrib = [0.6, 0.3, 0.2, 0.1]
        idx = 0
        while spend > 0:
            price = get_price(mean_score[idx][0])
            s = self.invest_amount*distrib[idx]
            s = math.ceil(s/price)
            spend -= s*price
            self.stock_rank[mean_score[idx][0]] = {'decision': s}

    # EXECUTION===================================================================================================
    #execute buy/sell suggestions from get_rank
    def execute(self, dict_rank):

        for stock, value in dict_rank.items():
            
            if stock not in list(self.holdings['Name']): # add new stock
                self.holdings = pd_append(self.holdings, {'Name': stock, 'Qnt': 0, 'UCost': 0, 'BaseCost': 0, 
                    'Price': 0, 'Value': 0, 'LongGain': 0, 'ShortGain': 0})

            stock_idx = self.holdings.loc[self.holdings['Name'] == stock].index[0]
            price = self.holdings.at[stock_idx, 'Price'] = get_price(stock)

            if value['decision'] > 0: #buy
                asset_value = price*value['decision']
                if asset_value > self.balance.at[0, 'Cash']: # check balance
                    print("Not enough money")
                    continue
                else:
                    self.activity = pd_append(self.activity, {'Name': stock, 'Type': 'buy', 'Time': date.today().strftime(DATE_FORMAT), 
                        'Qnt': value['decision'], 'Value': asset_value})
                    self.holdings.at[stock_idx, 'BaseCost'] += asset_value
                    self.holdings.at[stock_idx, 'Qnt'] += value['decision']
                    self.balance.at[0, 'Cash'] -= asset_value

            elif value['decision'] < 0: #sell
                # TODO: get oldest qnt and reduce it
                holding_qnt = self.holdings.at[stock_idx, 'Qnt']
                qnt = abs(value['decision']) if abs(value['decision']) <= holding_qnt else holding_qnt
                asset_value = price*qnt
                self.activity = pd_append(self.activity, {'Name': stock, 'Type': 'sell', 'Time': date.today().strftime(DATE_FORMAT), 
                    'Qnt': qnt, 'Value': asset_value})
                self.holdings.at[stock_idx, 'BaseCost'] -= asset_value
                self.holdings.at[stock_idx, 'Qnt'] -= qnt
                self.balance.at[0, 'Cash'] += asset_value
                
        # update portifolio
        self.update_all()
        return



    #update calculated numbers in files after execute 
    def update_all(self):
        # check portifolio numbers
        for index, row in self.holdings.iterrows():
            name = row['Name']
            self.holdings.at[index, 'Price'] = get_price(name)
            holding_qnt = self.holdings.at[index, 'Qnt']
            self.holdings.at[index, 'UCost'] = self.holdings.at[index, 'BaseCost']/holding_qnt if holding_qnt > 0 else 0 # UCost=BaseCost/Qnt
            self.holdings.at[index, 'Value'] = self.holdings.at[index, 'Price']*holding_qnt# Value=Price*Qnt
            # check taxtime Qnt ShortGain or LongGain
            dtoday = datetime.today()
            activity_index = self.activity[(self.activity['Type'] == 'buy') & (self.activity['Name'] == name)]
            LongGain, ShortGain = 0, 0
            for _, rowt in activity_index.iterrows():
                d1 = datetime.strptime(rowt['Time'], DATE_FORMAT)
                delta = dtoday - d1
                if delta.days > 365:
                    LongGain += rowt['Qnt'] 
                else:
                    ShortGain += rowt['Qnt']
            self.holdings.at[index, 'LongGain'] = LongGain
            self.holdings.at[index, 'ShortGain'] = ShortGain

        # check balance
        total_stock = self.holdings['Value'].sum()
        self.balance.at[0, 'Stock'] = total_stock
        self.balance.at[0, 'Time'] = str(datetime.now().strftime(DATE_FORMAT))



if __name__ == "__main__":
    args, config = get_arg()

    bot = TradingBot(config) # Create an instance of the trading bot
    today_date = datetime.today()
    prev_analysis_date = datetime.strptime(bot.balance['Time'].values[-1], DATE_FORMAT) #get last analysis date
    interval_date = today_date - prev_analysis_date

    if interval_date.days >= config['INTERVAL_ANALYSIS'] or args.mode == 'debug':
        bot.get_news() # run news checks
        bot.get_rank() # rank for suggestion
        print(bot.stock_rank)
        approve = input("Do you want to execute? (y/n)")
        if approve == 'y':
            bot.execute(bot.stock_rank)
        else:
            pass

    bot.save_back() # create backup
