## Bot to invest

Bot to help you choose what to invest using AI 

### Install

Install using pip
```
pip install -e .
```

### Run 
Go to folder `src/atradebot`

start by setting your plan using .yaml file like in  `default.yaml`: 


Then run `python3 main.py -c default.yaml -m run`
This will get news and suggest what stocks to buy during that time and update the profile `.xlsx` files

### Test strategies
Go to folder `src/atradebot`

Use this to test different strategies using past historical data
```
python backtest.py
```


### Set to run everyday
Go to folder `src/atradebot`

run: `crontab -e`

And add this line to run everyday 8am: 
`0 8 * * * python main.py -c default.yaml -m run`


### Run App

Use the following command to run this bot as an app

```
streamlit run app.py
```



### To do


user input:
- [x] balance (INIT_CASH)
- [x] time horizon (TIMEFRAME)

chrono job:
- [x] set to run analysis every month
- [x] set to update news every day

collect dynamic data:
- [x] check news every day
- [x] calculate sentiment for decision during specific dates
- [ ] correlate past history news with move backtesting

collect static data: check when need decision
- [ ] historical data 
- [ ] balance sheet

decision: given table of stocks or areas to invest 
- [x] news sentiment
- [ ] historic data info
- [ ] balance sheet
- [ ] personel (linkedin)
- [x] add backtesting

decision algorithm: rank best to buy and sell
- [x] simple average of news sentiment
- [ ] ask gpt
- [x] avg cost strategy

app
- [x] page1 QA
- [x] page2 stock strategy
- [ ] page3 chat

more stuff:
- [x] buy/hold
- [x] sell
- [x] add/swap to new stocks
- [ ] add bonds interest rate

output:
- [x] average-cost stat output profile files .xlsx
- [ ] email alert when to put money every month
- [x] backtest comparison graph
- [ ] graph of prediction
- [ ] optional: alpaca api to auto-execute trade


