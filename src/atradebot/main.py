
import asyncio
from datetime import datetime
from functools import partial
import pickle
from argparse import ArgumentParser
from datetime import datetime
import os
import json
from dataclasses import asdict, dataclass
from tqdm import tqdm
import yfinance as yf
import pandas as pd
import numpy as np
import traceback
import random

from atradebot.bots.bot import call_ai
from atradebot.bots.langchain_bot import TradingAgentsDebate
from atradebot.bots.prompts import *
from atradebot.utils.utils import *
from atradebot.utils.api import AlpacaAPI
from atradebot.config import *
from atradebot.rag.rag_system import TradingRAGSystem




def get_portfolio(portfolio):
    portfolio_str = '\n'.join([f"- {row['Ticker']} is currently priced at ${row['Price']:.2f}; The buying cost was ${row['Cost']:.2f}; Unit gain/loss is ${row['Price'] - row['Unit Cost']:.2f}; There are {row['Quantity']} shares" for _, row in portfolio.iterrows()])
    pstr = f'The current portfolio has following stocks: \n {portfolio_str} \n' if not portfolio.empty else ""
    return pstr

class Tradebot:
    def __init__(self, args):
        present_date = datetime.now().strftime(DATE_FORMAT)
        self.report_out = {'config':asdict(CONFIG),'present_date': present_date}
        self.args = args
        self.alpaca_api = AlpacaAPI(args.mode)
        self.portfolio = pd.DataFrame()
        self.available_cash = 0.0
        self.tickers = []

        self.trading_system = TradingAgentsDebate(
            max_debate_rounds=2, 
            use_local_llm=True,
            use_web_search=True
        )
        
    
    def load_portfolio(self):
        # get the current portfolio
        if self.args.portfolio_path:
            portfolio = pd.read_csv(self.args.portfolio_path) # columns: Ticker, Quantity, Cost, Unit Cost, Price  
            scale = random.randint(2, 10) 
            portfolio['Quantity'] = portfolio['Quantity'] / scale # scale down for testing
            available_cash = 10000 
            for idx, row in portfolio.iterrows():
                ticker = row['Ticker']
                info = yf.Ticker(ticker).info
                price = info.get('regularMarketOpen', 0)
                portfolio.at[idx, 'Price'] = price

        else:
            print("fetching portfolio from Alpaca API.")
            portfolio, available_cash = self.alpaca_api.alpaca_get_portfolio()
            available_cash = float(available_cash) # do dollar cost averaging over 6 mon. Buy once a week

        portfolio['PortfolioPercent'] = portfolio['Cost'] / portfolio['Cost'].sum()
        self.skip_tickers = [] # tickers to skip in research
        if FILTER and self.args.mode == 'prod':    
            portfolio = portfolio.sort_values(by='PortfolioPercent', ascending=False)
            self.skip_tickers = portfolio[portfolio['PortfolioPercent'] > 0.01]['Ticker'].tolist() # tickers to skip in research
            # filter out top percent positions in the portfolio for management
            portfolio = portfolio[portfolio['PortfolioPercent'] < 0.01]

        # portifolio is pandas DataFrame with at least the following columns: Ticker, Quantity, Cost, Unit Cost
        tickers = portfolio['Ticker'].tolist()
        tickers = list(set(tickers))  # Remove duplicates
        if len(tickers) == 0:
            print(f"No tickers to manage in the portfolio. Exiting.")

        self.tickers = tickers # tickers to check and rank
        self.portfolio = portfolio
        self.available_cash = available_cash

        return tickers, portfolio, available_cash
    
    def stock_research(self):
        """
        Perform stock research and provide a list of the best 5 growth stocks in the information technology sector.
        Args:
            sell (bool): If True, search for stocks to sell. Default is False.
            skip_tickers (list): List of stock tickers to skip. Default is None.
        Returns:
            tuple: A tuple containing the list of tickers for the recommended stocks and the research answer.

        Raises:
            Any exceptions that may occur during the execution.
        """    
        cur_date = datetime.now().strftime(DATE_FORMAT)
        # get news to search for stocks
        skip_str = ''
        if FILTER and self.skip_tickers: 
            skip_str = """
            Avoid your research and analysis on the stocks that are already in the current portfolio: {skip}.\n
            """
            skip_str = skip_str.format(skip=', '.join(self.skip_tickers))

        # example = "{'ticker': ['BIDU', 'TSM'], 'expected_price': [130, 679]}"
        # query = GLOBALSEARCH_PROMPT.format(curdate=cur_date, sector=CONFIG.SECTOR, example=example)
        example = "{'ticker': ['AAPL', 'MSFT'], 'expected_price': [235, 615]}"
        query = SEARCH_PROMPT.format(curdate=cur_date, sector=CONFIG.SECTOR, skip=skip_str)
       
        tickerlst = []
        if CONFIG.AGENT_MODE:
            print(f"Running Agent...")
            answer = self.trading_system.run_agent(query, SEARCH_PROMPT_OUTPUT.format(example=example))
        else:
            query += SEARCH_PROMPT_OUTPUT.format(example=example)
            answer = asyncio.run(call_ai(query, web_search=True, check_fn=partial(is_json_in_text, keys=['ticker'])))

        print(f"Stock Research Answer: {answer}...")

        if not is_json_in_text(answer, keys=['ticker']):
            print("No valid JSON found in the response.")
            self.report_out['stock_research_buy'] = answer
            return []
        stklist = find_json(answer)
        if not stklist:
            self.report_out['stock_research_buy'] = answer
            print("stklist not found.")
            return []
        
        tickerlst = [t for t in stklist.get('ticker',[]) if t in ALLTICKS]

        self.report_out['stock_research_buy'] = answer
        self.report_out['stock_research_summary'] = stklist
        self.report_out['stock_research_input'] = query

        if tickerlst:
            self.tickers.extend(tickerlst)
            self.tickers = list(set(self.tickers))

        return tickerlst

    def stock_manage(self, predict, rag=None):
        
        # predict rank
        news_pred = []
        for r in predict:
            ticker = r['ticker']
            analysis = r['answer']
            info = r['info']
            news_pred.append({'ticker': ticker, 'analysis': analysis, 'info': info})
        ticker_rank_str = '\n'.join([f"{r['ticker']}: {r['price_change']:.2f}" for r in predict])

        # explain news
        if rag:
            news_str = rag
        else:
            news_str = '\n'.join([f"{n['ticker']}: {n['analysis']}" for n in news_pred])


        example = {"AAPL": 2, "UNH": -4, "GOOGL": 0}  # Example allocation dictionary

        if self.available_cash:
            available_cash_str = f'Currently, the portfolio has ${self.available_cash:.2f} in cash.'
        else:
            available_cash_str = ''

        pstr = get_portfolio(self.portfolio) 
        prompt = MANAGE_PROMPT.format(news_str=news_str, 
                                    pstr=pstr, 
                                    ticker_rank_str=ticker_rank_str, 
                                    available_cash=available_cash_str)
        allocation = {}

        if CONFIG.AGENT_MODE:
            print(f"Running Agent...")
            msg = self.trading_system.run_agent(prompt, MANAGE_PROMPT_OUTPUT.format(example=str(example)))
        else:
            prompt += MANAGE_PROMPT_OUTPUT.format(example=str(example))
            msg = asyncio.run(call_ai(prompt, web_search=True, check_fn=is_json_in_text))
        allocation = find_json(msg)

        if self.args.execute:
            print(f"Executing trades based on the allocation {allocation}.")
            self.alpaca_api.alpaca_execute(allocation)
        
        self.portfolio = portfolio_update(allocation, self.portfolio)

        self.report_out['manage_input'] = prompt
        self.report_out['portfolio'] = self.portfolio.to_json()
        self.report_out['allocation'] = allocation

    def stock_check(self, ticker:str, ticker_data=None):
        """
        Perform stock research for a stock.
        Args:
            ticker (str): Stock ticker to research.
        Returns:
            tuple: A tuple containing the list of tickers for the recommended stocks and the research answer.

        Raises:
            Any exceptions that may occur during the execution.
        """    
        cur_date = datetime.now().strftime(DATE_FORMAT)
        if not ticker_data:
            info = yf.Ticker(ticker).info
        else:
            info = ticker_data.info

        stock_description = info.get('longBusinessSummary', '')
        current_price = info.get('regularMarketOpen', 0)

        example = '{"future_price_in_3_months": 100, "future_price_in_6_months": 75, "future_price_in_1_year": 120}'
        query = CHECK_PROMPT.format(stock=ticker, description=stock_description, curdate=cur_date, curprice=current_price)

        if CONFIG.AGENT_MODE:
            print(f"Running Agent for stock check on {ticker}.")
            answer = self.trading_system.run_agent(query, CHECK_PROMPT_OUTPUT.format(example=example))
        else:
            query += CHECK_PROMPT_OUTPUT.format(example=example)
            answer = asyncio.run(call_ai(query, web_search=True, check_fn=partial(is_json_in_text, keys=[FUTURE])))

        pricechanges = find_json(answer)
        if not is_json_in_text(answer, keys=[FUTURE]):
            return answer, info, query, {FUTURE: info['regularMarketOpen']} #if fail then just keep same price

        return answer, info, query, pricechanges

    def stock_rank(self):
        answers = []
        check_queries = []

        for ticker in tqdm(self.tickers):
            try:
                answer, info, check_query, price_changes = self.stock_check(ticker)
                print(f"Ticker: {ticker}, Analysis: {answer}")
                if FUTURE not in price_changes:
                    print(f"Price change for {FUTURE} not found, skipping ticker {ticker}.")
                    continue
                percent = price_changes[FUTURE] / info['regularMarketOpen']
                answers.append({'ticker': ticker, 'answer': answer, 'price_change': percent, 'info': info})
                check_queries.append(check_query)
            except Exception as e:
                print(f"Error processing ticker {ticker}: {e}")
                traceback.print_exc()
                
        
        # Sort by price change percentage
        answers.sort(key=lambda x: x['price_change'], reverse=True)
        self.report_out['stock_rank_input'] = check_queries
        self.report_out['stock_rank'] = answers
        return answers





def get_args(raw_args=None):
    parser = ArgumentParser(description="parameters")
    parser.add_argument('--portfolio_path','-p', type=str, default='', help='path to the portfolio CSV file')
    parser.add_argument('--execute','-e', action='store_true', help='execute the trades based on the portfolio')
    parser.add_argument('--provider', type=str, default='lmstudio', choices=['openai', 'lmstudio'], 
                       help='LLM provider to use (overrides config.py setting)')
    parser.add_argument('--mode', type=str, default='llm', choices=['agent', 'llm'], help='mode to run the bot in')
    parser.add_argument('--rag', action='store_true', help='enable RAG (Retrieval-Augmented Generation) for historical analysis')
    args = parser.parse_args(raw_args)

    # Override LLM provider if specified via command line
    if args.provider:
        original_provider = CONFIG.LLM_PROVIDER
        CONFIG.LLM_PROVIDER = args.provider
        print(f"Using LLM provider: {args.provider} (overriding config default: {original_provider})")
    else:
        print(f"Using LLM provider: {CONFIG.LLM_PROVIDER} (from config)")

    return args


# Agent steps: 
# search 
# RAG 
# rank 
# manage
def main():
    args = get_args()
    assert args.mode in ['llm', 'agent'], "mode must be one of 'llm' or 'agent'"

    present_date = datetime.now().strftime(DATE_FORMAT)

    print(f"Running in {args.mode} mode with paper trading.")

    if args.mode == 'agent':
        CONFIG.AGENT_MODE = True

    tb = Tradebot(args)

    asyncio.run(tb.trading_system._create_trading_graph())
    print("Searching for stocks...")

    tb.load_portfolio()

    # 1- SEARCH =================================================
    # search new stocks: => list of new stocks 
    stklist = tb.stock_research()

    # 2- RAG =================================================
    # Initialize RAG system for historical analysis
    # collect news and analysis for each stock (time period) => list of news and analysis for each stock
    rag_result = ''
    if args.rag:
        rag = TradingRAGSystem()
        rag.load_all_documents()
        rag_result = rag.get_stock_explain(tb.tickers+stklist)

    # 3- RANK ===========================================
    # explain each stock => prediction of price change in 6 months and rank

    print("Running stock predict+ranking...")
    predict = tb.stock_rank()
    # 4- MANAGE =================================================
    # manage have stocks: news+prediction+portfolio => buy/sell/hold allocation

    print("Running stock management...")
    tb.stock_manage(predict=predict, rag=rag_result)

    # save the report
    report_out = tb.report_out
    fn = f'{args.mode}_{args.provider}_{str(present_date)}_manage.pkl'
    with open(os.path.join(OUTPUT_D, fn), 'wb') as f:
        pickle.dump(report_out, f)
        print(f"Saved stock management report to {f.name}")


if __name__ == "__main__":
    main()

