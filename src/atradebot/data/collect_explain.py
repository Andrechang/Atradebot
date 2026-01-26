#!/usr/bin/env python3
"""
Stock Data Collection using explaination events. For RAG
"""

import json
import os
from datetime import datetime, timedelta
import json
import time
from atradebot.config import *
import asyncio
import pandas as pd

from atradebot.utils.utils import get_existing_tickers
from atradebot.utils.explainutils import explain, get_explain_summary

async def stock_collect(tickers, lookback_months=12):
    """
    Collect news and analysis for each stock.
    """
    list_fewshots = []
    present_date = datetime.now().strftime(DATE_FORMAT)

    for stock in tickers:
        print(f"Collecting explanations for {stock}...")
        past_date = (datetime.now() - pd.DateOffset(months=lookback_months)).strftime(DATE_FORMAT) # lookback months ago
        explanations, event_dates, data = await explain(stock, past_date, present_date, movingaverage=3)
        explain_str = get_explain_summary(explanations, stock)
        list_fewshots.append({'ticker': stock, 
                                'explanations': explanations, 
                                'summary': explain_str, 
                                'event_dates': event_dates, 
                                })
        

    return list_fewshots



async def run_collect_explain(gettickers:list):
    # select some stocks to gather news
    # Check existing stocks in HF_STOCKDB and remove from portfolio if already present
    filepath = HF_STOCKDB # used in rag_system.py -> stocks.py
    existing_stocks = get_existing_tickers(HF_STOCKDB)
    # Remove existing stocks from tickers
    gettickers = [stock for stock in gettickers if stock not in existing_stocks]
    print(f"Collecting explain for {len(gettickers)} stocks after filtering existing entries.")
    print("Running stock explain collection...")  
    news = await stock_collect(gettickers, lookback_months=18) # run using local model (costly for cloud models)
    with open(filepath, "a") as file:  # Open in append mode
        for entry in news:
            file.write(json.dumps(entry) + "\n")
            
    print('All done!')
    return news
    

if __name__ == "__main__":
    asyncio.run(run_collect_explain(['VERI', 'WEAV', 'META']))

