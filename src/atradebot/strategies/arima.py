import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from atradebot.data.yf_data import get_yf_data_some
from atradebot.strategies import register_strategy
from atradebot.strategies.strategy import Strategy, StrategyConfig
import os
from atradebot.utils.utils import find_business_day
import pickle
from pmdarima import auto_arima

@register_strategy("ARIMAStrategy")
class ARIMAStrategy(Strategy):
    """
    A class representing an ARIMA strategy for stock trading.

    Attributes:
        args (StrategyConfig): The configuration parameters for the strategy.
        train_past_date (datetime): The past date used for training the ARIMA models.
        stock_data (dict): A dictionary containing stock price data for each stock in the strategy's portfolio.
    
    Methods:
        predict(present_date, future_days, stock): Predicts the future stock prices using ARIMA models.
        generate_allocation(date, portfolio): Generates the allocation of stocks to buy or sell based on the predicted price changes.
    """    
    def __init__(self, args: StrategyConfig):
        """
        Initialize the Strategy object with the given arguments.

        Args:
            args (StrategyConfig): A configuration object containing necessary parameters.

        Returns:
            None

        Raises:
            None
        """        
        self.args = args
        # train needs around 2-5 years of data
        self.train_past_date = find_business_day(self.args.past_date, -self.args.train_days)
        self.stock_data = {}
        for stock in self.args.stocks: # get historical data for each stock
            data = get_yf_data_some(self.args.stocks, self.train_past_date, self.args.past_date)
            stock_prices = data['Close']
            stock_prices.index = pd.to_datetime(stock_prices.index)  # Ensure datetime format
            stock_prices = stock_prices.asfreq('B').fillna(method='ffill')
            self.stock_data[stock] = stock_prices

    def predict(self, present_date, future_days, stock):
        """
        Predict future stock prices using ARIMA model.

        Args:
            present_date (str): Date for which prediction is being made.
            future_days (int): Number of days into the future to predict.
            stock (str): Stock symbol for which prediction is being made.

        Returns:
            tuple: A tuple containing the log dictionary, percentage change in price, and predicted price.
        """        
        past_date = find_business_day(present_date, -self.args.inference_days)
        data = get_yf_data_some(stock, past_date, present_date)
        stock_prices = data['Close']
        stock_prices.index = pd.to_datetime(stock_prices.index)  # Ensure datetime format
        stock_prices = stock_prices.asfreq('B').fillna(method='ffill')

        best_arima = auto_arima(stock_prices, seasonal=False, trace=True, suppress_warnings=True) # Fit best ARIMA model
        arima_model = ARIMA(stock_prices, order=best_arima.order)
        arima_fit = arima_model.fit()
        price = arima_fit.forecast(steps=future_days)
        pred_pct = (price.iloc[-1] - stock_prices.iloc[-1]) / stock_prices.iloc[-1]
        log = {}
        return log, pred_pct[-1], price[-1]

    def generate_allocation(self, present_date, portfolio):
        """
        Generate allocation for a portfolio based on future price predictions.

        Args:
            date (str): The date for which allocation is generated.
            portfolio (Portfolio): The portfolio object containing information about stocks.

        Returns:
            dict: A dictionary mapping stocks to their corresponding allocation values.
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
    

if __name__ == "__main__":
    args = StrategyConfig(
        inference_days=20,
        future_days=10,
        train_days=365*2,
        past_date="2024-10-10",
        present_date="2024-11-21",
        stocks=['AAPL']
    )
    strategy = ARIMAStrategy(args)
    a, pred, price = strategy.predict(args.present_date, 10, "AAPL")
    print(pred, price)

