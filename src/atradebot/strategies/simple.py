from pypfopt import risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.efficient_frontier import EfficientFrontier
from atradebot.strategies.strategy import Strategy, StrategyConfig
from atradebot.strategies import register_strategy
from atradebot.utils.utils import find_business_day
from atradebot.data.yf_data import get_yf_data_some
from atradebot.params import DATE_FORMAT
from datetime import datetime
import random
# Define a simple strategy that allocates 50% of the portfolio on the first day
def max_sharpe_allocation(data, amount_invest):
    """
    Calculate the maximum Sharpe ratio allocation for a given amount to invest.

    Args:
        data (DataFrame): Historical price data of assets.
        amount_invest (float): Total amount to invest.

    Returns:
        tuple: Allocation dictionary of assets and leftover cash.

    Raises:
        None
    """    
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    ef = EfficientFrontier(mu, S)
    raw_weights = ef.max_sharpe(risk_free_rate=-10)
    cleaned_weights = ef.clean_weights()
    latest_prices = get_latest_prices(data)
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=amount_invest)
    allocation, leftover = da.greedy_portfolio()
    return allocation, leftover

@register_strategy("SimpleStrategy")
class SimpleStrategy(Strategy):
    """
    A class representing a simple strategy for trading stocks.

    Attributes:
        args (StrategyConfig): The strategy configuration object containing necessary parameters for the strategy.
        stocks (dict): A dictionary with stock symbols as keys and the number of shares as values.
        cash (list): A list of cash amounts allocated for trading.
        cash_idx (int): An index indicating the current cash amount being used.
        days_interval (int): The number of days interval for cash allocation.
        prev_date (datetime.date): The previous date considered for allocation.
        data (pandas.Series): The stock price data for trading.
    """    
    def __init__(self, args:StrategyConfig):
        """
        Create simple strategy based on sharpe allocation

        Parameters:
            - past_date (datetime.date): date start to backtest in format yyyy-mm-dd
            - present_date (datetime.date): date end to backtest in format yyyy-mm-dd
            - cash (int, optional): amount to invest, defaults to 10000
        """      
        self.args = args          
        cash=args.cash
        self.cash = [cash*0.5, cash*0.3, cash*0.1, cash*0.1] #amount to invest in each rebalance
        self.cash_idx = 0
        delta = self.args.present_date - self.args.past_date
        self.days_interval = delta.days/len(self.cash) #rebalance 4 times
        self.prev_date = self.args.past_date  #previous date for rebalance
    
    def predict(self, present_date, future_days, stock):
        """
        Predict the future stock price based on historical data.

        Args:
            present_date (datetime): The present date for prediction.
            future_days (int): Number of days into the future for prediction.
            stock (str): Stock symbol for prediction.

        Returns:
            tuple: A tuple containing the log dictionary, prediction value, and predicted stock price.

        Note:
            This function relies on external functions find_business_day and get_yf_data_some for data retrieval.
        """        
        past_date = find_business_day(present_date, -self.args.inference_days)
        stock_data = get_yf_data_some(stock, past_date, present_date)
        pred = random.uniform(-1, 1)
        price_ = stock_data['Close'].iloc[-1] * pred
        log = {}
        return log, pred, price_

    def generate_allocation(self, date, portfolio):
        """
        Generate allocation based on the date.
        Args:
            date (pandas.Timestamp): Date to generate allocation.
            portfolio (pandas.DataFrame): Current portfolio (check backtest.py for format).
        Returns:
            dict: Allocation for each stock.
        """        
        delta = date.date() - self.prev_date
        data = portfolio.price_data['Close']
        idx = data.index.get_loc(str(date.date()))
        if date.date() == self.prev_date: #first day
            allocation, leftover = max_sharpe_allocation(data[0:idx], amount_invest=self.cash[self.cash_idx])
            self.cash_idx += 1                        
            return allocation
        elif delta.days > self.days_interval: #rebalance 
            allocation, leftover = max_sharpe_allocation(data[0:idx], amount_invest=self.cash[self.cash_idx])
            self.prev_date = date.date()
            self.cash_idx += 1
            return allocation
        else:
            return {}


if __name__ == "__main__":
    args = StrategyConfig()
    f = SimpleStrategy(args)
    present_date = datetime.strptime("2024-12-10", DATE_FORMAT)
    f.predict(present_date, 10, "AAPL")
