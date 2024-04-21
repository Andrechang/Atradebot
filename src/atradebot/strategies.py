# Add different strategies modules here
# Strategy: needs one function that will be called by the backtester
# Strategy must have function: generate_allocation with:
# inputs: date, portfolio
# outputs: dict of allocations for each stock

from pypfopt import risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.efficient_frontier import EfficientFrontier
import re
import torch
import numpy as np
from atradebot.utils import DATE_FORMAT
from atradebot import fin_train, utils, utils_news
import yfinance as yf

device = "cuda" if torch.cuda.is_available() else "cpu"


# Define a simple strategy that allocates 50% of the portfolio on the first day
def max_sharpe_allocation(data, amount_invest):
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    ef = EfficientFrontier(mu, S)
    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    latest_prices = get_latest_prices(data)
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=amount_invest)
    allocation, leftover = da.greedy_portfolio()
    return allocation, leftover



class SimpleStrategy:
    def __init__(self, start_date, end_date, data, stocks, cash=10000):
        """Create simple strategy based on sharpe allocation

        :param start_date: date start to backtest in format yyyy-mm-dd
        :type start_date: datetime.date
        :param end_date: date end to backtest in format yyyy-mm-dd
        :type end_date: datetime.date
        :param data: stock price data with train date and eval date. data[train_start_data:eval_end_date]
        :type data: pandas.DataFrame
        :param stocks: list of stocks to trade
        :type stocks: List[str]
        :param cash: amount to invest, defaults to 10000
        :type cash: int, optional
        """        

        self.prev_date = start_date  #previous date for rebalance
        self.stocks = {i: 0 for i in stocks}
        
        self.cash = [cash*0.5, cash*0.3, cash*0.1, cash*0.1] #amount to invest in each rebalance
        self.cash_idx = 0

        self.start_date = start_date
        self.end_date = end_date
        delta = end_date - start_date
        self.days_interval = delta.days/len(self.cash) #rebalance 4 times
        self.data = data['Adj Close']

    def generate_allocation(self, date, portfolio):
        """Generate allocation based on the date
        :param date: date to generate allocation
        :type date: pandas.Timestamp
        :param portfolio: current portfolio (check backtest.py for format)
        :type portfolio: pandas.DataFrame dict{stock: number of shares}
        :return: allocation for each stock
        :rtype: dict{"stock": number of stocks to buy/sell }
        """        
        delta = date.date() - self.prev_date
        idx = self.data.index.get_loc(str(date.date()))
        if date.date() == self.prev_date: #first day
            allocation, leftover = max_sharpe_allocation(self.data[0:idx], amount_invest=self.cash[self.cash_idx])
            self.cash_idx += 1                        
            return allocation
        elif delta.days > self.days_interval: #rebalance 
            allocation, leftover = max_sharpe_allocation(self.data[0:idx], amount_invest=self.cash[self.cash_idx])
            self.prev_date = date.date()
            self.cash_idx += 1
            return allocation
        else:
            return self.stocks



class FinForecastStrategy:
    def __init__(self, start_date, end_date, data, stocks, cash=10000, model_id="fin_forecast_0"):
        """
        data: stock_forecast_0 generated from fin_data.generate_forecast_task
        model: trained using fin_train.py
        model input: 
            Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                ## Instruction: what is the forecast for ... 
                ## Input: date, news title, news snippet
        model output:
                ## Response: forecast percentage change for 1mon, 5mon, 1 yr

        :param start_date: date start to backtest in format yyyy-mm-dd
        :type start_date: datetime.date
        :param end_date: date end to backtest in format yyyy-mm-dd
        :type end_date: datetime.date
        :param data: pandas dataframe from yfinance
        :type data: pandas dataframe
        :param stocks: list of stocks to backtest
        :type stocks: List[str]
        :param cash: amount of cash to invest, defaults to 10000
        :type cash: int, optional
        :param model_id: huggingface model id to use, defaults to "fin_forecast_0"
        :type model_id: str, optional
        """        

        self.prev_date = start_date #previous date for rebalance
        self.stocks = {i: 0 for i in stocks}
        
        self.cash = [cash*0.5, cash*0.3, cash*0.1, cash*0.1] #amount to invest in each rebalance
        self.cash_idx = 0

        self.start_date = start_date
        self.end_date = end_date
        delta = end_date - start_date
        self.days_interval = delta.days/len(self.cash) #rebalance 4 times
        self.data = data['Adj Close']

        self.model, self.tokenizer = fin_train.get_lrg_model(model_id)
        self.model.to(device)

    def model_run(self, date, num_news=5):
        """get news before date and forecast using model

        :param date: date = pd.Timestamp("2021-07-22")
        :type date: pandas.Timestamp
        :param num_news: number of news to get, defaults to 5
        :type num_news: int, optional
        :return: prediction for each stock. Prediction format [1 mon, 5 mon, 1 yr]
        :rtype: dict{stock: list of pred }
        """        
        
        end = date.date()
        start = utils.business_days(end, -5)

        all_stocks = {}
        for stock in self.stocks:
            news = utils_news.get_news(stock=stock, time_period=[start, end], num_results=num_news, news_source='google')
            if not news:
                print(f"No news for {stock} on dates: {start} to {end}. Source: google")
                continue
            all_pred = []
            for new in news:
                in_dict = {'instruction': f"what is the forecast for {new['stock']}", 
                        'input':f"{new['date']} {new['title']} {new['snippet']}"}
                prompt = fin_train.generate_prompt(in_dict, mode='eval')
                in_data = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length")
                in_data['input_ids'] = in_data['input_ids'].to(device)
                with torch.cuda.amp.autocast():
                    outputs = self.model.generate(input_ids = in_data['input_ids'], 
                        attention_mask = in_data['attention_mask'],
                        max_new_tokens=128,
                        pad_token_id= self.tokenizer.eos_token_id,
                        eos_token_id= self.tokenizer.eos_token_id
                    )

                response = fin_train.get_response(outputs[0].cpu().numpy(), self.tokenizer)
                pred = re.findall(r"[-+]?(?:\d*\.*\d+)", response)
                print(f"{new['stock']} forecast: {response} \n {pred}")
                if len(pred) > 3:
                    pred = pred[:3]
                if len(pred) < 3:
                    continue
                
                pred = [eval(i) for i in pred] #convert to str to float
                all_pred.append(pred)

            #average forecast
            avg_pred = np.mean(np.array(all_pred), axis=0)
            all_stocks[stock] = avg_pred
        return all_stocks


    def model_allocation(self, date, amount_invest, prices, portfolio, sell_mode=False):
        """get allocation for each stock based on model predictions

        :param date: date = pd.Timestamp("2021-07-22")
        :type date: pandas.Timestamp
        :param amount_invest: amount to invest
        :type amount_invest: int
        :param prices: list of stock prices
        :type prices: pandas.Series
        :param portfolio: current portfolio (check backtest.py for format)
        :type portfolio: pandas.DataFrame dict{stock: number of shares}
        :param sell_mode: whether to sell stocks, defaults to False
        :type sell_mode: bool, optional
        :return: allocation for each stock
        :rtype: dict{stock: number to buy/sell}
        """             
        all_stocks = self.model_run(date)
        #pick top increasing forecast
        future_mode = 0 #choose timeline 1mon, 5mon, 1yr
        alloc = sorted(all_stocks.items(), key=lambda x: x[1][future_mode], reverse=True) #most increase first 
        # alloc: tuple(stock, forecast)
        weights = [0.6, 0.3, 0.1] #weight for each stock
        amounts = [int(amount_invest * weight) for weight in weights]
        allocation = {i[0]: int(amounts[idx]/prices[i[0]]) for idx, i in enumerate(alloc[:3])}#get top3 stocks

        #get top stocks to sell
        if sell_mode:
            alloc_sell = sorted(all_stocks.items(), key=lambda x: x[1][future_mode], reverse=False) #most decrease first 
            weights_sell = 0.5 #weight to sell holdings 
            for sell in alloc_sell:
                if portfolio[sell[0]][date] > 0:
                    amounts_sell = int(portfolio[sell[0]][date]*weights_sell)
                    allocation[sell[0]] = -amounts_sell
                    break

        leftover = amount_invest - sum([prices[i[0]]*allocation[i[0]] for i in alloc[:3]])
        return allocation, leftover
        #TODO balance with max_sharpe_allocation
        

    def generate_allocation(self, date, portfolio):
        """generate allocation for each stock using average cost investment method

        :param date: date = pd.Timestamp("2021-07-22")
        :type date: pandas.Timestamp
        :param portfolio: current portfolio (check backtest.py for format)
        :type portfolio: pandas.DataFrame dict{stock: number of shares}
        :return: allocation for each stock
        :rtype: dict{stock: number to buy/sell}
        """               
        delta = date.date() - self.prev_date
        idx = self.data.index.get_loc(str(date.date()))
        if date.date() == self.prev_date: #first day
            allocation, leftover = self.model_allocation(date, amount_invest=self.cash[self.cash_idx], 
                prices=self.data.iloc[idx], portfolio=portfolio, sell_mode=False)
            self.cash_idx += 1                        
            return allocation
        elif delta.days > self.days_interval: #rebalance 
            allocation, leftover = self.model_allocation(date, amount_invest=self.cash[self.cash_idx], 
                prices=self.data.iloc[idx], portfolio=portfolio, sell_mode=True)
            self.prev_date = date.date()
            self.cash_idx += 1
            return allocation
        else:
            return self.stocks



class FinOneStockStrategy:
    def __init__(self, start_date, end_date, data, stocks, cash=10000, model_id="fin_gpt2_one_nvda", news_src="google"):
        """
        data: stocks_one_nvda generated from fin_data.generate_onestock_task
        model: trained using fin_train.py
        model input: 
            Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                ## Instruction: I have {n} {AAPL} stocks and {x} cash to invest. Given the recent news, should I buy, sell or hold {AAPL} stocks ? 
                ## Input: date, news snippet
        model output:
                ## Response: allocation suggestion

        :param start_date: date start to backtest in format yyyy-mm-dd
        :type start_date: datetime.date
        :param end_date: date end to backtest in format yyyy-mm-dd
        :type end_date: datetime.date
        :param data: pandas dataframe from yfinance
        :type data: pandas dataframe
        :param stocks: list of stocks to backtest
        :type stocks: List[str]
        :param cash: amount of cash to invest, defaults to 10000
        :type cash: int, optional
        :param model_id: huggingface model id to use, defaults to "fin_forecast_0"
        :type model_id: str, optional
        """        
        self.prev_date = start_date #previous date for rebalance
        self.stocks = {i: 0 for i in stocks}
        
        self.cash = [cash*0.5, cash*0.3, cash*0.1, cash*0.1] #amount to invest in each rebalance
        self.cash_idx = 0

        self.start_date = start_date
        self.end_date = end_date
        delta = end_date - start_date
        self.days_interval = delta.days/len(self.cash) #rebalance 4 times
        self.data = data['Adj Close']

        # self.model, self.tokenizer = fin_train.get_slm_model(model_id)
        self.model, self.tokenizer = fin_train.get_lrg_model(model_id)
        self.model.to(device)
        self.news_src = news_src

    def model_run(self, date, portfolio, prices, num_news=3, amount_invest=10000):
        """get news before date and forecast using model

        :param date: date = pd.Timestamp("2021-07-22")
        :type date: pandas.Timestamp
        :param portfolio: current portfolio (check backtest.py for format)
        :type portfolio: pandas.DataFrame dict{stock: number of shares}
        :param prices: list of stock prices
        :type prices: pandas.Series
        :param num_news: number of news to get, defaults to 5
        :type num_news: int, optional
        :param amount_invest: amount to invest
        :type amount_invest: int
        :return: prediction for each stock. Prediction format [1 mon, 5 mon, 1 yr]
        :rtype: dict{stock: list of pred }
        """        
        
        end = date.date()
        start = utils.business_days(end, -5) # also get news 5 days before
        all_stocks = {}
        for stock in self.stocks:
            corp_info = yf.Ticker(stock).info
            stock_price = prices[stock]
            #get news
            news = utils_news.get_news(stock, [start, end], num_news, self.news_src)
            if not news:
                print(f"No news for {stock} on dates: {start} to {end}. Source: {self.news_src}")
                continue

            #generate prompt
            txt = '' # collect only parts of news that references the stock
            query = f"{stock} {corp_info['longName']} {corp_info['longBusinessSummary']}"
            for new in news:
                txt += utils.get_doc2vectext(query, new['text'])

            #generate output based on allocation
            in_dict = {
                'instruction':f"I have {amount_invest} cash to invest. \
                {stock} price now is {stock_price} and given the recent news, should I buy, sell or hold {stock} stocks ? ",
                'input':f"News from {end}, {txt}", 
                }
            if portfolio[stock][date] > 0:
                in_dict['instruction'] = f"I have {portfolio[stock][date]} {stock} stocks and {amount_invest} cash to invest. \
                    {stock} price now is {stock_price} and given the recent news, should I buy, sell or hold {stock} stocks ? "
            prompt = fin_train.generate_prompt(in_dict, mode='eval')
            
            #run model
            in_data = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            in_data['input_ids'] = in_data['input_ids'].to(device)
            in_data['attention_mask'] = in_data['attention_mask'].to(device)
            with torch.cuda.amp.autocast():
                outputs = self.model.generate(input_ids = in_data['input_ids'], 
                    attention_mask = in_data['attention_mask'],
                    max_new_tokens=128,
                    pad_token_id= self.tokenizer.eos_token_id,
                    eos_token_id= self.tokenizer.eos_token_id
                )

            response = fin_train.get_response(outputs[0].cpu().numpy(), self.tokenizer)
            print(f"on date {end} suggestion for {stock} is: {response} \n")
            
            nnum = re.findall(r'\d+', response)
            if nnum and nnum[0].isnumeric():
                nnum = int(nnum[0])
            if 'buy' in response.lower():
                all_stocks[stock] = int(nnum)
            elif 'sell' in response.lower():
                all_stocks[stock] = -int(nnum)
            else: #hold
                all_stocks[stock] = 0

        return all_stocks


    def model_allocation(self, date, prices, portfolio, amount_invest=10000):
        """get allocation for each stock based on model predictions

        :param date: date = pd.Timestamp("2021-07-22")
        :type date: pandas.Timestamp
        :param prices: list of stock prices
        :type prices: pandas.Series
        :param portfolio: current portfolio (check backtest.py for format)
        :type portfolio: pandas.DataFrame dict{stock: number of shares}
        :param amount_invest: amount to invest
        :type amount_invest: int
        :return: allocation for each stock
        :rtype: dict{stock: number to buy/sell}
        """             
        allocation = self.model_run(date, portfolio, prices, num_news=3, amount_invest=amount_invest)
        for stock in allocation:
            max_stocks = amount_invest/prices[stock]
            allocation[stock] = min(allocation[stock], max_stocks)

        leftover = portfolio['Cash'] - sum([prices[stock]*allocation[stock] for stock in allocation])
        return allocation, leftover
        

    def generate_allocation(self, date, portfolio):
        """generate allocation for each stock using average cost investment method

        :param date: date = pd.Timestamp("2021-07-22")
        :type date: pandas.Timestamp
        :param portfolio: current portfolio (check backtest.py for format)
        :type portfolio: pandas.DataFrame dict{stock: number of shares}
        :return: allocation for each stock
        :rtype: dict{stock: number to buy/sell}
        """               
        delta = date.date() - self.prev_date
        idx = self.data.index.get_loc(str(date.date()))
        if date.date() == self.prev_date: #first day
            allocation, leftover = self.model_allocation(date, amount_invest=self.cash[self.cash_idx], 
                prices=self.data.iloc[idx], portfolio=portfolio)
            self.cash_idx += 1                        
            return allocation
        elif delta.days > self.days_interval: #rebalance 
            allocation, leftover = self.model_allocation(date, amount_invest=self.cash[self.cash_idx], 
                prices=self.data.iloc[idx], portfolio=portfolio)
            self.prev_date = date.date()
            self.cash_idx += 1
            return allocation
        else:
            return self.stocks
            