from abc import ABC, abstractmethod
import datetime
from dataclasses import dataclass, asdict, fields
import pandas

@dataclass
class StrategyConfig:
    # arguments for strategy
    """
    A dataclass representing the configuration for a trading strategy.

    Attributes:
        data (pandas.DataFrame): The historical data for trading.
        past_date (datetime.date): The starting date for historical data.
        present_date (datetime.date): The current date for trading.
        stocks (list): The list of stocks to trade.
        inference_days (int): Number of days to consider for inference.
        future_days (int): Number of days to predict into the future.
        train_days (int): Number of days to train the model. history-data used in arima and strategies that train as it goes [Rolling Forecast]
        cash (int): Initial amount of capital for trading.
        newsapi (str): API to use for news data.
        strategy (str): The trading strategy to use.
    """    
    past_date: datetime.date = datetime.date(2023, 5, 1)
    present_date: datetime.date = datetime.date(2023, 12, 10)
    inference_days: int = 10
    future_days: int = 10
    train_days: int = 365*3  
    newsapi: str = 'dataset'
    strategy: str = 'SimpleStrategy'
    run_trade: bool = False
    cash: int = 10000
    
    

class Strategy(ABC):
    
    @abstractmethod   
    def predict(self, present_date:datetime.date, future_days:int, stock:str):
        # returns: model output, percentage change, predicted price
        """
        Predict the stock price for a given future date.

        Args:
            present_date (datetime.date): The present date for which the prediction is made.
            future_days (int): The number of days in the future for which the prediction is made.
            stock (str): The stock symbol for which the prediction is made.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented in a subclass.
        """        
        pass # Predict function not implemented

    @abstractmethod
    def generate_allocation(self, date:datetime.date, portfolio):
        # returns: allocation: dict{"stock": number of stocks to buy/sell }
        """
        Generate allocation for a specific date and portfolio.

        Args:
            date (datetime.date): The date for which allocation needs to be generated.
            portfolio (object): The portfolio for which allocation needs to be generated.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """        
        pass # Generate allocation function not implemented