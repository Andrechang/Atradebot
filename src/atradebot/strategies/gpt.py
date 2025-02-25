# Description: GPT-based strategies

import torch
import re
from openai import OpenAI
from atradebot.params import OPENAI_API, DATE_FORMAT, ALLTICKS
from datetime import datetime
from atradebot.data.yf_data import get_yf_data_some, get_yf_data_all
from atradebot.data.llm_prompt import LLM_INSTRUCTION, construct_prompt
from atradebot.utils.utils import find_business_day, extract_last_number
from atradebot.strategies import register_strategy
from atradebot.strategies.strategy import Strategy, StrategyConfig

OPENAI_CLIENT = OpenAI(api_key=OPENAI_API)

@register_strategy("GPTStrategy")
class GPTStrategy(Strategy): 
    """
    A class representing a GPTStrategy for generating stock allocation.

    Attributes:
        inference_days (int): The number of days used for prediction inference.
        future_days (int): The number of days into the future to predict.
        newsapi (NewsAPI): The API used for fetching news data.
        client (OpenAI): The OpenAI client for accessing the GPT model.
    """    
    def __init__(self, args:StrategyConfig):
        """
        Initialize the Strategy object with the specified configuration.

        Args:
            args (StrategyConfig): An object containing the configuration settings.

        Returns:
            None

        Attributes:
            inference_days (int): Number of days to use for inference.
            future_days (int): Number of days to predict into the future.
            newsapi (str): API key for accessing news data.
            client (OpenAI): Client object for accessing OpenAI API.
        """        
        self.inference_days = args.inference_days
        self.future_days = args.future_days
        self.newsapi = args.newsapi
        self.client = OPENAI_CLIENT
        # NOTE: the model may already know the current stock price, so cant backtest it
    
    def generate_allocation(self, present_date:datetime.date, portfolio):
        """
        Generate allocation for the given portfolio based on future price predictions.

        Args:
            present_date (datetime.date): The date for which allocation is to be generated.
            portfolio: An object representing the portfolio containing tickers.

        Returns:
            dict: A dictionary with stock tickers as keys and allocation quantity as values.
        """        
        future_price, allocation = {}, {}
        for stock in portfolio.tickers:
            _, pred_pct, _ = self.predict(present_date, self.future_days, stock)
            future_price[stock] = pred_pct
        # buy top5 and sell bottom5
        top5 = sorted(future_price.items(), key=lambda x: x[1], reverse=True)[:5]
        bot5 = sorted(future_price.items(), key=lambda x: x[1], reverse=False)[:5]
        buy_qnt = 5; sell_qnt = 5
        for stock, pred in top5:
            if pred > 0:
                allocation[stock] = buy_qnt
                buy_qnt -= 1
        for stock, pred in bot5:
            if pred < 0:
                allocation[stock] = -sell_qnt
                sell_qnt -= 1
        return allocation
    
    def predict(self, present_date:datetime.date, future_days:int, stock:str): 
        """
        Predict the future stock price based on past data and GPT-4o model.

        Args:
            present_date (datetime.date): The current date for which the prediction is being made.
            future_days (int): Number of days in the future to predict the stock price.
            stock (str): The stock symbol for which the prediction is being made.

        Returns:
            tuple: A tuple containing the log dictionary, predicted percentage change, and predicted future stock price.

        Raises:
            None
        """        
        past_date = find_business_day(present_date, -self.inference_days)
        stock_data = get_yf_data_some(stock, past_date, present_date)
        prompt, news = construct_prompt(stock, present_date, past_date, futuredays=future_days, newsapi=self.newsapi)

        completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": LLM_INSTRUCTION},
                    {"role": "user", "content": prompt}])
        message = completion.choices[0].message.content
        answer = message.split('%')[0]
        pred = extract_last_number(answer)
        torch.cuda.empty_cache()
        cur_price = stock_data['Close'].iloc[-1, -1]
        if pred is None:
            print("Failed to get prediction from the model.")
            return None, 0, cur_price
        
        price_ = cur_price*(pred/100) + cur_price
        log = {"prompt": prompt, "answer": answer, "news": news.to_dict()}
        return log, pred, price_  
    
if __name__ == "__main__":
    args = StrategyConfig()
    f = GPTStrategy(args)
    present_date = datetime.strptime("2024-12-10", DATE_FORMAT)
    a, pred, price = f.predict(present_date, 10, "AAPL")
    print(pred, price)