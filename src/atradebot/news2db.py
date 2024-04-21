# get news on stocks and run sentiment
# save stock, date, sentiment

# import json
import datetime
import dateparser
import sqlalchemy as db
from db_stocks import create_db
from sentiment_utils import get_sentiment, sentiment_analyzer
from utils import DATE_FORMAT
from news_utils import get_google_news
from stocks_lists import stocks_list_all


def get_news_sentiments_store_db(stocks_list, from_date, to_date, num_results=10, verbose=False):
	# setup db:
	engine, connection, stocks, dates, news, sentiments = create_db()
	
	for stock in stocks_list:
		stock_news, _, _ = get_google_news(stock, 
				      num_results=num_results, 
					  time_period=[from_date, to_date]
					)
		print('Articles #:', len(stock_news))
		print(stock_news)

		# # save to json so we do not have to get news again:
		# with open("stock_news.json", "w") as text_file:
		#     jsonString = json.dumps(stock_news)
		#     text_file.write(jsonString)
		#     text_file.close()

		# read json
		# jsonString = open("stock_news.json", "r").read()

		# add articles to database:
		# for article in json.loads(jsonString):
		for article in stock_news:
			# print(article)
			# convert date:
			converted_date = dateparser.parse(article['date'])
			# print(article['date'], converted_date)
			# run sentiment analysis:
			txt_sentiment = get_sentiment(article['text'], sentiment_analyzer, max_length=512)
			# print(txt_sentiment)

			# add to news:
			entry = [{'symbol':article['stock'], 
					'title':article['title'], 
					'date':converted_date, 
					'url':article['link'], 
					'source':article['source'], 
					'text':article['text'], 
					'sentiment':txt_sentiment, 
					'embedding':'[0,0,0]'},
					]
			query = db.insert(news)
			ResultProxy = connection.execute(query,entry)

			#add to sentiments:
			entry = [{'symbol':article['stock'], 
					'date':converted_date, 
					'sentiment':txt_sentiment,} 
					]
			query = db.insert(sentiments)
			ResultProxy = connection.execute(query,entry)


	if verbose:
		# query news:
		print('\n\nNEWS DB:\n')
		query = news.select()
		ResultProxy = connection.execute(query)
		ResultSet = ResultProxy.fetchall()
		print(ResultSet)

		# query sentiments:
		print('\n\SENTIMENTS DB:\n')
		query = sentiments.select()
		ResultProxy = connection.execute(query)
		ResultSet = ResultProxy.fetchall()
		print(ResultSet)

# script to run CRON job - get news for all stocks in our list
if __name__ == "__main__":
	# stocks_list = ['AAPL']
	stocks_list = stocks_list_all # get our list of stocks
	date_today = datetime.date.today().strftime(DATE_FORMAT)
	# run sentimet over last 7days (1 week), max 10 articles per symbol
	get_news_sentiments_store_db(stocks_list, 
				  from_date=(datetime.datetime.now() - datetime.timedelta(days=7)).strftime(DATE_FORMAT), 
				  to_date=date_today,
			      num_results=10,
				  verbose=True,
				)