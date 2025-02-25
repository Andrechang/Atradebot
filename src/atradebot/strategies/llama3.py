# based on: https://github.com/AI4Finance-Foundation/FinGPT/tree/master/fingpt/FinGPT_Forecaster
# Huggingface Llama3 based serving


import torch
from peft import PeftModel
from collections import defaultdict
from datetime import date, datetime, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from atradebot.data.yf_data import get_yf_data_some
from atradebot.params import DATE_FORMAT
from atradebot.strategies import register_strategy
from atradebot.strategies.strategy import Strategy
from atradebot.utils.utils import find_business_day, extract_last_number
from atradebot.data.llm_prompt import construct_prompt, format_chat_template
from atradebot.trainer.train_llama3 import BASEMODEL, LORAMODEL, RESPONSE_TOKENS
from atradebot.strategies.strategy import StrategyConfig

@register_strategy("FinLlama3Strategy")
class FinLlama3Strategy(Strategy): 
    """
    A class representing a financial trading strategy using language model inference.

    Attributes:
        newsapi (NewsAPI): The NewsAPI instance used to retrieve news for the strategy.
        inference_days (int): The number of days of historical data to use for model inference.
        future_days (int): The number of days into the future to predict stock prices.
        model (AutoModelForCausalLM): The pretrained language model for generating predictions.
        tokenizer (AutoTokenizer): The tokenizer for processing input text.
        streamer (TextStreamer): The TextStreamer instance for handling text data streams.
    """    
    def __init__(self, args:StrategyConfig):
        """
        Initialize the Strategy object with the given arguments.

        Args:
            args (StrategyConfig): Object containing configuration parameters for the strategy.

        Returns:
            None

        Raises:
            None
        """        
        self.newsapi = args.newsapi
        self.inference_days = args.inference_days
        self.future_days = args.future_days
        self.model = AutoModelForCausalLM.from_pretrained(
            BASEMODEL,
            trust_remote_code=True, 
            device_map="auto",
            torch_dtype=torch.float16)
        # self.model = PeftModel.from_pretrained(self.bmodel,LORAMODEL)
        self.model = self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(BASEMODEL)
        self.streamer = TextStreamer(self.tokenizer)
    
    def generate_allocation(self, present_date:datetime.date, portfolio):
        """
        Generate allocation for a portfolio based on predicted future prices.

        Args:
            date (datetime.date): The date for which the allocation is generated.
            portfolio (object): An object representing the portfolio containing stock tickers.

        Returns:
            dict: A dictionary containing the allocation for each stock ticker.

        Raises:
            None
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
        # return: log: dict with model input and output to save, pred: prediction, price_: predicted price
        """
        Predict the future stock price using a language model.

        Args:
            present_date (datetime.date): The present date for which the prediction is being made.
            future_days (int): Number of future days for which the prediction is needed.
            stock (str): Stock symbol for which the prediction is being made.

        Returns:
            tuple: A tuple containing log information, prediction percentage, and the predicted price.

        Raises:
            None
        """        
        past_date = find_business_day(present_date, -self.inference_days)
        stock_data = get_yf_data_some(stock, past_date, present_date)
        prompt, news = construct_prompt(stock, present_date, past_date, futuredays=future_days, newsapi=self.newsapi)
        row = {"instruction": prompt}
        prompt = format_chat_template(self.tokenizer, row)

        inputs = self.tokenizer(prompt["text"], return_tensors='pt', padding=False)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}            
        res = self.model.generate(
            **inputs, max_length=128_000, do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True, 
            # streamer=self.streamer
        )
        output = self.tokenizer.decode(res[0])
        ii = output.find(RESPONSE_TOKENS)
        output = output[ii:]
        print('OUTPUT: ', output)
        answer = output.split('%')[0]
        pred = extract_last_number(answer)
        torch.cuda.empty_cache()
        cur_price = stock_data['Close'].iloc[-1][-1]
        if pred is None:
            print("Failed to get prediction from the model.")
            return None, 0, cur_price
        
        price_ = cur_price*(pred/100) + cur_price
        log = {"prompt": prompt, "answer": answer, "news": news}
        return log, pred, price_        
        
        
# example answer:
if __name__ == "__main__":
    args = StrategyConfig()
    f = FinLlama3Strategy(args)
    present_date = datetime.strptime("2024-12-10", DATE_FORMAT)
    a, pred, price = f.predict(present_date, 10, "AAPL")
    print(pred, price)