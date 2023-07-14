# get news on stocks and run sentiment
# save stock, date, sentiment

import json
import dateparser
import sqlalchemy as db
from db import create_db
from sentiment_utils import get_sentiment, sentiment_analyzer
from news_utils import get_google_news

# setup db:
engine, connection, stocks, dates, news, sentiments = create_db()

# get some news for stock list:
stocks_list = ['AAPL']

for stock in stocks_list:
	# stock_news, _, _ = get_google_news(stock, num_results=10, time_period=['2022-06-28', '2022-06-31'])
	# print('Articles #:', len(stock_news))
	# print(stock_news)

	# # save to json so we do not have to get news again:
	# with open("stock_news.json", "w") as text_file:
	#     jsonString = json.dumps(stock_news)
	#     text_file.write(jsonString)
	#     text_file.close()

	# read json
	jsonString = open("stock_news.json", "r").read()

	# add articles to database:
	for article in json.loads(jsonString):
		# convert date:
		converted_date = dateparser.parse(article['date'])

		print(article['date'], converted_date)
		# print(article)
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


# # query news:
# query = news.select()
# ResultProxy = connection.execute(query)
# ResultSet = ResultProxy.fetchall()
# print(ResultSet)


# query sentiments:
query = sentiments.select()
ResultProxy = connection.execute(query)
ResultSet = ResultProxy.fetchall()
print(ResultSet)