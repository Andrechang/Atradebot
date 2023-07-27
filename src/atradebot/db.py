# E. Culurciello
# July 2023

# create a database of stocks, news, embeddings, sentiment
# each stock and each date can have multiple news

import os

import requests
import sqlalchemy as db
import yfinance as yf
import pandas as pd
from sqlalchemy import text


def create_db():
    # create database:
    engine = db.create_engine('sqlite:///atradebot.db', echo=True)
    connection = engine.connect()
    metadata = db.MetaData()

    # create tables:
    stocks = db.Table('stocks', metadata,
                        db.Column('id', db.Integer(), primary_key=True),
                        db.Column('symbol', db.String(255), nullable=True),
                        db.Column('name', db.String(255), nullable=True),
                        db.Column('sector', db.String(255), nullable=True),
                        db.Column('industry', db.String(255), nullable=True),
                        db.Column('live_price', db.Float(), nullable=True),
                        db.Column('prev_close', db.Float(), nullable=True),
                        db.Column('open', db.Float(), nullable=True),
                    )

    dates = db.Table('dates', metadata,
                        db.Column('id', db.Integer(), primary_key=True),
                        db.Column('date', db.String(255), nullable=False),
                    )

    news = db.Table('news', metadata,
                        db.Column('id', db.Integer(), primary_key=True),
                        db.Column('symbol', db.String(255), nullable=False),
                        db.Column('title', db.String(255), nullable=False),
                        db.Column('date', db.String(255), nullable=False),
                        db.Column('url', db.String(255), nullable=False),
                        db.Column('source', db.String(255), nullable=False),
                        db.Column('text', db.String(255), nullable=False),
                        db.Column('sentiment', db.Float(), nullable=False),
                        db.Column('embedding', db.String(255), nullable=False),
                    )

    sentiments = db.Table('sentiments', metadata,
                        db.Column('id', db.Integer(), primary_key=True),
                        db.Column('symbol', db.String(255), nullable=False),
                        db.Column('date', db.String(255), nullable=False),
                        db.Column('sentiment', db.Float(), nullable=False),
                    )

    # create tables in database:
    print("creating tables")
    metadata.create_all(engine)
    print("tables created")
    return engine, connection, stocks, dates, news, sentiments


if __name__ == "__main__":
    print("creating database")
    engine, connection, stocks, dates, news, sentiments = create_db()
    print("database created")

    connection.execute(text("PRAGMA journal_mode=DELETE"))

    # get list of stocks:
    stock_df = pd.read_excel('S&P 500 Companies.xlsx')
    symbols = stock_df['Symbol'].tolist()

    # get data for symbols:
    for i in symbols:
        try:
            ticker = yf.Ticker(i)

            # get stock info
            info = ticker.info

            name = info.get('shortName', 'NA')
            sector = info.get('sector', 'NA')
            industry = info.get('industry', 'NA')
            live_price = info.get('regularMarketPrice', 0.0)
            prev_close = info.get('previousClose', 0.0)
            open_price = info.get('open', 0.0)

            if live_price is None:
                live_price = 0.0
            if industry is None:
                industry = ""
            if sector is None:
                sector = ""
            if name is None:
                name = ""
            if prev_close is None:
                prev_close = 0.0
            if open_price is None:
                open_price = 0.0


            # insert stock data into tables:
            query = db.insert(stocks)
            values_list = [{'symbol':i, 'name':name, 'sector':sector, 'industry':industry, 'live_price':live_price, 'prev_close':prev_close, 'open':open_price}]
            print("inserting stock data for: ", i)
            ResultProxy = connection.execute(query,values_list)
            print("inserted stock data for: ", i)

        # Error handling
        except requests.exceptions.HTTPError as error_terminal:
            print(f"HTTPError occurred while getting data for {i}: {error_terminal}")
            continue
        except Exception as e:  # General exception catch
            print(f"Unexpected error occurred while getting data for {i}: {e}")
            continue

    ResultProxy = connection.execute(db.select(stocks))
    ResultSet = ResultProxy.fetchall()

    # print the first 5 rows:
    for row in ResultSet[:5]:
        print("This is Row")
        print(row)

    # insert news
    query = db.insert(news)
    values_list = [{'symbol':'AAPL', 'title':'Apple Inc. (AAPL) Stock Sinks As Market Gains: What You Should Know', 'date':'2021-07-23 21:50:09', 'url':'https://finance.yahoo.com/news/apple-inc-aapl-stock-sinks-015009416.html', 'source':'Yahoo Finance', 'text':'Apple Inc. (AAPL) closed at $148.56 in the latest trading session, marking a -0.47% move from the prior day.', 'sentiment':0.5, 'embedding':'0.1,0.2,0.3'},
                ]
    ResultProxy = connection.execute(query,values_list)

    # query data:
    query = stocks.select()
    ResultProxy = connection.execute(query)
    ResultSet = ResultProxy.fetchall()
    print("stocks quered")

    # query data:
    query = news.select()
    ResultProxy = connection.execute(query)
    ResultSet = ResultProxy.fetchall()

    # check query data:
    res = connection.execute(db.select(stocks))

    rows = res.fetchmany(10)

    for r in rows:
        print(r)

    connection.execute(text("PRAGMA journal_mode=WAL"))

    # Close the connection
    connection.close()
