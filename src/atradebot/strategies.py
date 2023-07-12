# Add different strategies modules here
# Strategy: needs one function that will be called by the backtester
# Strategy must have function: generate_allocation with:
# inputs: date, portfolio
# outputs: dict of allocations for each stock

from datetime import date, datetime
from pypfopt import risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.efficient_frontier import EfficientFrontier
import re
from atradebot.utils import business_days
from atradebot.news_util import get_google_news, get_finhub_news
from atradebot.fin_train import get_model, generate_prompt
import torch
import numpy as np
from atradebot.fin_train import get_response

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
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
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
        Args:
            date (pandas.Timestamp): date to generate allocation

        Returns:
            dict{"stock": number of stocks to buy/sell }: allocation for each stock
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
    def __init__(self, start_date, end_date, data, stocks, cash=10000, model_id="achang/fin_forecast"):
        """
        model:
        data: achang/stock_forecast
        input: 
            Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                ## Instruction: what is the forecast for ... 
                ## Input: date, news title, news snippet
        output:
                ## Response: forecast percentage change for 1mon, 5mon, 1 yr
        """

        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        self.prev_date = start_date #previous date for rebalance
        self.stocks = {i: 0 for i in stocks}
        
        self.cash = [cash*0.5, cash*0.3, cash*0.1, cash*0.1] #amount to invest in each rebalance
        self.cash_idx = 0

        self.start_date = start_date
        self.end_date = end_date
        delta = end_date - start_date
        self.days_interval = delta.days/len(self.cash) #rebalance 4 times
        self.data = data['Adj Close']

        self.model, self.tokenizer = get_model(model_id)
        self.model.to(device)

    def model_run(self, date, num_news=5):
        """get news before date and forecast using model

        Args:
            date (pandas.Timestamp): date = pd.Timestamp("2021-07-22")
            num_news (int, optional): number of news to get. Defaults to 5.

        Returns:
            dict{stock: list of pred }: prediction for each stock. Prediction format [1 mon, 5 mon, 1 yr]
        """

        end = date.date()
        start = business_days(end, -5)
        all_stocks = {}
        for stock in self.stocks:
            news, _, _ = get_google_news(stock=stock, num_results=num_news, time_period=[start, end])
            assert len(news) > 0, "no news found, google search blocked error"
            all_pred = []
            for new in news:
                in_dict = {'instruction': f"what is the forecast for {new['stock']}", 
                        'input':f"{new['date']} {new['title']} {new['snippet']}"}
                prompt = generate_prompt(in_dict, mode='eval')
                in_data = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length")
                in_data['input_ids'] = in_data['input_ids'].to(device)
                with torch.cuda.amp.autocast():
                    outputs = self.model.generate(input_ids = in_data['input_ids'], 
                        attention_mask = in_data['attention_mask'],
                        max_new_tokens=128,
                        pad_token_id= self.tokenizer.eos_token_id,
                        eos_token_id= self.tokenizer.eos_token_id
                    )

                response = get_response(outputs[0].cpu().numpy(), self.tokenizer)
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
        Args:
            date (pandas.Timestamp): date = pd.Timestamp("2021-07-22")
            amount_invest (int): amount to invest
            prices (pandas.Series): list of stock prices
            portfolio (dict{stock: number of shares}): current portfolio
            sell_mode (bool, optional): whether to sell stocks. Defaults to False.
        Returns:
            dict{stock: number to buy/sell}: allocation for each stock
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
        Args:
            date (pandas.Timestamp): date = pd.Timestamp("2021-07-22")

        Returns:
            dict{stock: number to buy/sell}: allocation for each stock
        """        
        delta = date.date() - self.prev_date
        idx = self.data.index.get_loc(str(date.date()))
        if date.date() == self.prev_date: #first day
            # allocation, leftover = max_sharpe_allocation(self.data[0:idx], amount_invest=self.cash[self.cash_idx])
            allocation, leftover = self.model_allocation(date, amount_invest=self.cash[self.cash_idx], 
                prices=self.data.iloc[idx], portfolio=portfolio, sell_mode=False)
            self.cash_idx += 1                        
            return allocation
        elif delta.days > self.days_interval: #rebalance 
            # allocation, leftover = max_sharpe_allocation(self.data[0:idx], amount_invest=self.cash[self.cash_idx])
            allocation, leftover = self.model_allocation(date, amount_invest=self.cash[self.cash_idx], 
                prices=self.data.iloc[idx], portfolio=portfolio, sell_mode=True)
            self.prev_date = date.date()
            self.cash_idx += 1
            return allocation
        else:
            return self.stocks