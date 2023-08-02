import os
import requests
import sqlalchemy as db
import yfinance as yf
import pandas as pd
from sqlalchemy import text
import time
from ib_insync import *

def create_db():
    # create database:
    engine = db.create_engine('sqlite:///atradebot.db', echo=True)
    connection = engine.connect()
    metadata = db.MetaData()

    # create a single table:
    data = db.Table('data', metadata,
                        db.Column('id', db.Integer(), primary_key=True),
                        db.Column('symbol', db.String(255), nullable=True),
                        db.Column('name', db.String(255), nullable=True),
                        db.Column('sector', db.String(255), nullable=True),
                        db.Column('industry', db.String(255), nullable=True),
                        db.Column('live_price', db.Float(), nullable=True),
                        db.Column('prev_close', db.Float(), nullable=True),
                        db.Column('open', db.Float(), nullable=True),
                        db.Column('title', db.String(255), nullable=True),
                        db.Column('news_date', db.String(255), nullable=True),
                        db.Column('url', db.String(255), nullable=True),
                        db.Column('source', db.String(255), nullable=True),
                        db.Column('text', db.String(255), nullable=True),
                    )

    # create table in database:
    metadata.create_all(engine)
    return engine, connection, data

if __name__ == "__main__":
    engine, connection, data = create_db()

    connection.execute(text("PRAGMA journal_mode=DELETE"))

    # get list of stocks:
    stock_df = pd.read_excel('src/atradebot/SP_500_Companies.xlsx')
    symbols = stock_df['Symbol'].tolist()

    ib = IB()
    ib.connect('127.0.0.1', 7496, clientId=1)
    news_providers = ib.reqNewsProviders()
    codes = '+'.join(news_provider.code for news_provider in news_providers)

    # get data for symbols:
    for i in symbols[:5]:
        trans = connection.begin_nested()
        try:
            values_list = {'symbol': i}

            # get stock info
            try:
                ticker = yf.Ticker(i)
                info = ticker.info

                values_list.update({
                    'name': info.get('shortName', 'NA'),
                    'sector': info.get('sector', 'NA'),
                    'industry': info.get('industry', 'NA'),
                    'live_price': info.get('regularMarketPrice', 0.0),
                    'prev_close': info.get('previousClose', 0.0),
                    'open': info.get('open', 0.0),
                })

            # Error handling for stock info fetching
            except requests.exceptions.HTTPError as error_terminal:
                print("HTTPError while fetching stock info for symbol:", i)
            except Exception as e:  # General exception catch
                print("Unexpected error while fetching stock info for symbol:", i)

            # Fetch and store news articles
            try:
                stock = Stock(i, 'SMART', 'USD')
                ib.qualifyContracts(stock)
                headlines = ib.reqHistoricalNews(stock.conId, codes, '', '', 100)

                for headline in headlines:
                    article_date = headline.time.date()
                    article = ib.reqNewsArticle(headline.providerCode, headline.articleId)

                    # Insert the article into the database
                    values_list.update({
                        'title': '',  # Title not needed
                        'news_date': str(article_date),
                        'url': '',  # URL not provided
                        'source': '',  # Source not provided
                        'text': article.articleText
                    })

            # Error handling for news fetching
            except requests.exceptions.HTTPError as error_terminal:
                print("HTTPError while fetching news for symbol:", i)
            except Exception as e:  # General exception catch
                print("Unexpected error while fetching news for symbol:", i)

            # Insert data into table:
            query = db.insert(data)
            ResultProxy = connection.execute(query, [values_list])

            trans.commit()
            time.sleep(1)

        # Error handling
        except Exception as e:  # General exception catch
            print("Unexpected error:", e)
            trans.rollback()

    # Fetch and print the first 5 stocks from the database after processing
    #query = db.select([data]).limit(5)
    query = data.select().limit(5)
    ResultProxy = connection.execute(query)
    ResultSet = ResultProxy.fetchall()

    for result in ResultSet:
        print(result)

    connection.execute(text("PRAGMA journal_mode=WAL"))

    # Close the connection
    connection.close()
