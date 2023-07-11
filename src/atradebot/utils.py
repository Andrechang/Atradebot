
from datetime import date, datetime, timedelta
import yaml
import requests
from bs4 import BeautifulSoup
import trafilatura
from trafilatura.settings import use_config

trafilatura_config = use_config()
trafilatura_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")

DATE_FORMAT = "%Y-%m-%d"


def is_business_day(date):
    return bool(len(pd.bdate_range(date, date)))

#add/subtract business days
def business_days(start_date, num_days):
    current_date = start_date
    business_days_added = 0
    while business_days_added < abs(num_days):
        if num_days < 0:
            current_date -= timedelta(days=1)
        else:
            current_date += timedelta(days=1)
        if current_date.weekday() < 5:  # Monday to Friday
            business_days_added += 1
    return current_date


def pd_append(df, dict_d):
    return pd.concat([df, pd.DataFrame.from_records([dict_d])])

def get_price(stock):
    ticker = yf.Ticker(stock).info
    return ticker['regularMarketOpen']



# holdings: Name, Qnt, UCost (unit cost), BaseCost, Price (current price), Value (current Value), LongGain (Qnt), ShortGain (Qnt)
# activity: Name, type (buy/sell), TB (time bought), Qnt, Proceed
# balance: Time, Cash, Stock, Total
# news: Time, Name, Text, Score, Link
def get_config(cfg_file):
    with open(cfg_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


# run financial sentiment analysis model
def get_sentiment(text, model, max_length=512):
    sentences = tokenize.sent_tokenize(text)
    # truncate sentences that are too long
    for i, s in enumerate(sentences):
        if len(s) > max_length:
            sentences[i] = sentences[i][:max_length]

    sentiment = model(sentences)
    sum, neutrals = 0, 0
    if len(sentiment) > 0:
        for r in sentiment: 
            sum += (r["label"] == "Positive")
            neutrals += (r["label"] == "Neutral")

        den = len(sentiment)-neutrals
        sentiment = sum/den if den > 0 else 1.0 # as all neutral
    return sentiment


def get_forecast(stock, date):
    """get forecast for stock on date
    Args:
        stock (string): stock id
        date (string): date format 'Feb 2, 2019' "%b %d, %Y"

    Returns:
        forecast: forecast list [1mon, 5mon, 1 yr], higher than 1. percent increase, lower than 1. percent decrease
    """    
    s = datetime.strptime(date, "%b %d, %Y")
    e = business_days(s, +3)
    data = yf.Ticker(stock)
    hdata = data.history(start=s.strftime("%Y-%m-%d"),  end=e.strftime("%Y-%m-%d"))
    price = hdata['Close'].mean()

    forecast = [0, 0, 0]
    add_days = [21, 5*21, 12*21] #add business days
    for idx, adays in enumerate(add_days):
        s = business_days(s, adays)#look into future
        e = business_days(s, +3)
        hdata = data.history(start=s.strftime("%Y-%m-%d"),  end=e.strftime("%Y-%m-%d"))
        forecast[idx] = hdata['Close'].mean()/price
    
    return forecast

# collect google search text 
def get_google_news(stock, num_results=10, time_period=[]):
    # time_period=['2019-06-28' (start_time), '2019-06-29' (end_time)]
    query = stock
    headers = {
            "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36"
        }
    if time_period:
        query += f"+before%3A{time_period[1]}+after%3A{time_period[0]}" # add time range     
    search_req = "https://www.google.com/search?q="+query+"&gl=us&tbm=nws&num="+str(num_results)+""
    #https://developers.google.com/custom-search/docs/xml_results#WebSearch_Request_Format
    news_results = []
    # get webpage content
    response = requests.get(search_req, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    for el in soup.select("div.SoaBEf"):
        sublink = el.find("a")["href"]
        downloaded = trafilatura.fetch_url(sublink)
        html_text = trafilatura.extract(downloaded, config=trafilatura_config)
        if html_text:
            news_results.append(
                {
                    "link": el.find("a")["href"],
                    "title": el.select_one("div.MBeuO").get_text(),
                    "snippet": el.select_one(".GI74Re").get_text(),
                    "date": el.select_one(".LfVVr").get_text(),
                    "source": el.select_one(".NUnG9d span").get_text(),
                    "text": html_text,
                    "stock": stock,
                }
            )
    return news_results, search_req, soup