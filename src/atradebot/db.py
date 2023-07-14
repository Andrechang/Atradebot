# E. Culurciello
# July 2023

# create a database of stocks, news, embeddings, sentiment
# each stock and each date can have multiple news

import os
import sqlalchemy as db


def create_db():
    # create database:
    engine = db.create_engine('sqlite:///atradebot.db', echo=True)
    connection = engine.connect()
    metadata = db.MetaData()

    # create tables:
    stocks = db.Table('stocks', metadata,
                        db.Column('id', db.Integer(), primary_key=True),
                        db.Column('symbol', db.String(255), nullable=False),
                        db.Column('name', db.String(255), nullable=False),
                        db.Column('sector', db.String(255), nullable=False),
                        db.Column('industry', db.String(255), nullable=False),
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
    metadata.create_all(engine)
    return engine, connection, stocks, dates, news, sentiments


if __name__ == "__main__":
    engine, connection, stocks, dates, news = create_db()

    # insert data into tables:
    query = db.insert(stocks)
    values_list = [{'symbol':'AAPL', 'name':'Apple Inc.', 'sector':'Technology', 'industry':'Consumer Electronics'},
                    {'symbol':'MSFT', 'name':'Microsoft Corporation', 'sector':'Technology', 'industry':'Software—Infrastructure'},
                    {'symbol':'AMZN', 'name':'Amazon.com, Inc.', 'sector':'Consumer Cyclical', 'industry':'Internet Retail'},
                    {'symbol':'GOOG', 'name':'Alphabet Inc.', 'sector':'Technology', 'industry':'Internet Content & Information'},
                    {'symbol':'FB', 'name':'Facebook, Inc.', 'sector':'Communication Services', 'industry':'Internet Content & Information'},
                    {'symbol':'TSLA', 'name':'Tesla, Inc.', 'sector':'Consumer Cyclical', 'industry':'Auto Manufacturers'},
                    {'symbol':'NVDA', 'name':'NVIDIA Corporation', 'sector':'Technology', 'industry':'Semiconductors'},
                ]
    ResultProxy = connection.execute(query,values_list)

    # insert news
    query = db.insert(news)
    values_list = [{'symbol':'AAPL', 'title':'Apple Inc. (AAPL) Stock Sinks As Market Gains: What You Should Know', 'date':'2021-07-23 21:50:09', 'url':'https://finance.yahoo.com/news/apple-inc-aapl-stock-sinks-015009416.html', 'source':'Yahoo Finance', 'text':'Apple Inc. (AAPL) closed at $148.56 in the latest trading session, marking a -0.47% move from the prior day.', 'sentiment':0.5, 'embedding':'0.1,0.2,0.3'},
                ]
    ResultProxy = connection.execute(query,values_list)

    # query data:
    query = stocks.select()
    ResultProxy = connection.execute(query)
    ResultSet = ResultProxy.fetchall()
    print(ResultSet)

    # query data:
    query = news.select()
    ResultProxy = connection.execute(query)
    ResultSet = ResultProxy.fetchall()
    print(ResultSet)