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
import time


from sqlalchemy import create_engine, MetaData, text
from llama_index import LLMPredictor, ServiceContext, SQLDatabase, VectorStoreIndex
from llama_index.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from langchain import OpenAI

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
    metadata.create_all(engine)
    return engine, connection, stocks, dates, news, sentiments


if __name__ == "__main__":
    engine, connection, stocks, dates, news, sentiments = create_db()

    connection.execute(text("PRAGMA journal_mode=DELETE"))

    # get list of stocks:
    stock_df = pd.read_excel('SP_500_Companies.xlsx')
    symbols = stock_df['Symbol'].tolist()

    try:
        # get data for symbols:
        for i in symbols:
            trans = connection.begin_nested()
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
                ResultProxy = connection.execute(query,values_list)

                trans.commit()
                time.sleep(1)

            # Error handling
            except requests.exceptions.HTTPError as error_terminal:
                print(error_terminal)
                trans.rollback()
                continue
            except Exception as e:  # General exception catch
                print(e)
                trans.rollback()
                continue

        # insert news
        trans_news = connection.begin_nested()
        try:
            query = db.insert(news)
            values_list = [{'symbol':'AAPL', 'title':'Apple Inc. (AAPL) Stock Sinks As Market Gains: What You Should Know', 'date':'2021-07-23 21:50:09', 'url':'https://finance.yahoo.com/news/apple-inc-aapl-stock-sinks-015009416.html', 'source':'Yahoo Finance', 'text':'Apple Inc. (AAPL) closed at $148.56 in the latest trading session, marking a -0.47% move from the prior day.', 'sentiment':0.5, 'embedding':'0.1,0.2,0.3'},]
            ResultProxy = connection.execute(query,values_list)
            trans_news.commit()
        except Exception as e:
            print(e)
            trans_news.rollback()

    finally:
        # query data:
        query = stocks.select()
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



def test_openai_emb():
    # OpenAI API key:
    # os.environ["OPEN_AI_KEY"] = "your_api_key"

    # set up database:
    url = "sqlite:///atradebot.db"
    db = create_engine(url)

    # connection:
    conn = db.connect()

    # Print first 5 rows of stocks table to check connection:
    # # SQL query:
    # sql_query = text("SELECT * FROM stocks LIMIT 5;")
    #
    # try:
    #     results = conn.execute(sql_query)
    #     rows = results.fetchall()  # Fetch all rows from the result
    # except Exception as e:
    #     print(f"Error executing query: {e}")
    #     rows = None
    #
    # if rows:
    #     for row in rows:
    #         print(row)
    # else:
    #     print("No results returned from query")
    #
    # conn.close()

    # loading table definitions:
    metadata_obj = MetaData()
    metadata_obj.reflect(db)

    # print(metadata_obj.tables.keys())

    # create a database object:
    db_obj = SQLDatabase(db)

    # table node mapping:
    table_node_mapping = SQLTableNodeMapping(db_obj)

    table_schema_objs = []
    for table_name in metadata_obj.tables.keys():
        table_schema_objs.append(SQLTableSchema(table_name=table_name))

    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
    )


    # llm:
    llm = OpenAI(model_name="gpt-4", max_tokens=6000)

    # llm predictor:
    llm_predictor = LLMPredictor(llm=llm)

    # service context:
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    # query engine:
    query_engine = SQLTableRetrieverQueryEngine(
        db_obj,
        obj_index.as_retriever(similarity_top_k=1),
        service_context = service_context,
    )
