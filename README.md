## Bot to invest

start by setting your plan in `main.py`: 

```
TIMEFRAME = 1*365 # days: 1yrs 
INIT_CASH = 10000 # investment 10k over a year
INTERVAL_ANALYSIS = 15 # days to analyze and invest 
STOCKS2CHECK = ['AAPL','ABBV','AMZN','ASML','BHP','COST','GOOGL','JNJ','KLAC','LLY','LRCX','MSFT','NVDA','TSLA'] # list of stocks to check
```

Then run `python3 main.py -m init`
This will create files that will track news, stocks you invested, balance, time bought and sold in `.csv`

Then run `python3 main.py -m run`
This will get news and suggest what stocks to buy during that time and update the profile `.csv` files

### set to run everyday

run: `crontab -e`

And add this line to run everyday 8am: 
`0 8 * * * python main.py -m run`




