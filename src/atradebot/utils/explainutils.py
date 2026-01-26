import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from stockstash import TimeSeriesCache, ParquetStore, YFinanceProvider

from atradebot.config import *
from atradebot.bots.prompts import *
from atradebot.utils.utils import *
from atradebot.bots.bot import call_ai
from scipy.signal import find_peaks


def find_peaks_valleys(prices:np.array, prominence=1, distance=5):
    """
    Method 2: SciPy find_peaks - Most flexible and powerful
    
    Uses prominence (how much a peak stands out) and distance (minimum gap between peaks).
    This is generally the BEST method for stock data.
    
    Args:
        prominence: Minimum prominence (height above surrounding valleys). 
                    If None, auto-calculated as 1% of price range
        distance: Minimum number of points between peaks
        
    Returns:
        (peak_indices, valley_indices)
    """
    print(f"\n📈 Method: SciPy find_peaks (distance={distance})")
    
    # Auto-calculate prominence if not provided
    if prominence is None:
        price_range = prices.max() - prices.min()
        prominence = price_range * 0.01  # 1% of price range
        print(f"   Auto-calculated prominence: ${prominence:.2f}")
    
    # Find peaks
    peaks, peak_properties = find_peaks(
        prices, 
        prominence=prominence,
        distance=distance
    )
    
    # Find valleys (peaks in inverted data)
    valleys, valley_properties = find_peaks(
        -prices,
        prominence=prominence,
        distance=distance
    )
    
    print(f"   Found {len(peaks)} peaks and {len(valleys)} valleys")
    
    return peaks, valleys


def filter_events(data: np.array, peak_idx, valley_idx, method='topk', 
                 min_time_diff=10, min_price_diff=0.1, topk=7):
    """
    Unified filtering function for peaks and valleys.
    
    Args:
        data: Price data (numpy array or pandas Series)
        peak_idx: List of peak indices
        valley_idx: List of valley indices
        method: 'topk' for percentage-based filtering, 'distance' for time/price filtering
        min_time_diff: Minimum time difference between events (for distance method)
        min_price_diff: Minimum price difference as fraction (for distance method)
        topk: Number of top events to return (for topk method)
        
    Returns:
        tuple: (filtered_peak_idx, filtered_valley_idx)
    """
    if method == 'topk':
        # Filter based on percentage change and return top k events
        all_points = [(i, 'peak') for i in peak_idx] + [(i, 'valley') for i in valley_idx]
        all_points.sort(key=lambda x: x[0])  # Sort by index
        
        # Add last index as final event
        last_idx = len(data) - 1
        if not all_points or all_points[-1][0] != last_idx:
            all_points.append((last_idx, 'end'))
        
        # Calculate percentage changes between consecutive events
        events_with_pctchg = []
        for i in range(len(all_points) - 1):
            start_idx = all_points[i][0]
            end_idx = all_points[i + 1][0]
            
            start_price = data[start_idx]
            end_price = data[end_idx]
            
            pct_change = abs((end_price - start_price) / (start_price if start_price != 0 else 1e-8)) * 100
            
            events_with_pctchg.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_type': all_points[i][1],
                'pct_change': pct_change
            })
        
        # Sort by percentage change and select top k
        top_events = sorted(events_with_pctchg, key=lambda x: x['pct_change'], reverse=True)[:topk]
        
        # Extract indices for peaks and valleys from top events
        filtered_peaks = []
        filtered_valleys = []
        
        for event in top_events:
            start_idx = event['start_idx']
            start_type = event['start_type']
            
            if start_type == 'peak':
                filtered_peaks.append(start_idx)
            elif start_type == 'valley':
                filtered_valleys.append(start_idx)
        
        return filtered_peaks, filtered_valleys
    
    elif method == 'distance':
        # Filter based on time and price distance
        all_points = sorted([(i, 'peak') for i in peak_idx] + [(i, 'valley') for i in valley_idx])
        filtered = []

        last_idx = None
        last_price = None

        for idx, kind in all_points:
            if last_idx is not None:
                if abs(idx - last_idx) < min_time_diff:
                    continue
                price = data[idx]
                if last_price is not None and abs(price - last_price) / max(abs(last_price), 1e-8) < min_price_diff:
                    continue
            filtered.append((idx, kind))
            last_idx = idx
            last_price = data[idx]

        filtered_peaks = [idx for idx, kind in filtered if kind == 'peak']
        filtered_valleys = [idx for idx, kind in filtered if kind == 'valley']
        return filtered_peaks, filtered_valleys
    
    else:
        raise ValueError(f"Unknown filtering method: {method}")



def collect_events(stock=None, past_date=None, present_date=None, 
                    data=None, 
                    movingaverage=3, 
                    filter_method='topk', 
                    min_time_diff=10,
                    min_price_diff=0.1, 
                    topk=7):
    """
    Unified function to collect significant price movement events.

    Args:
        stock (str, optional): Stock symbol. Required if data is None.
        past_date (str, optional): Start date in yyyy-mm-dd format. Required if data is None.
        present_date (str, optional): End date in yyyy-mm-dd format. Required if data is None.
        data (DataFrame, optional): Pre-loaded price data. If None, will load from stock/dates.
        movingaverage (int): Window for moving average smoothing.
        filter_method (str): 'topk' for percentage-based filtering, 'distance' for time/price filtering.
        min_time_diff (int): Minimum time difference between events (for distance method).
        min_price_diff (float): Minimum price difference as fraction (for distance method).
        topk (int): Number of top events to return (for topk method).

    Returns:
        tuple: (events_dates, data) where events_dates is list of significant event dates
    """    
    # Load data if not provided
    if data is None:
        if not all([stock, past_date, present_date]):
            raise ValueError("Either provide data or stock/past_date/present_date parameters")
            
        manager = TimeSeriesCache(
            store=ParquetStore(OUTPUT_YFDATA),
            provider=YFinanceProvider()
        )
        data = manager.load(stock, past_date, present_date)

        if data.empty: 
            print(f"No data found for {stock} from {past_date} to {present_date}.")
            return [], data

    # Apply moving average smoothing
    data_stock = data['Close'].values.squeeze()
    data_mean = pd.Series(data_stock).rolling(window=movingaverage, center=True).mean().fillna(method='bfill').fillna(method='ffill').values

    # Find peaks and valleys
    peak_idx, valley_idx = find_peaks_valleys(data_mean)
    
    # Filter events based on specified method
    peak_idx, valley_idx = filter_events(
        data_mean, peak_idx, valley_idx, 
        method=filter_method,
        min_time_diff=min_time_diff,
        min_price_diff=min_price_diff,
        topk=topk
    )
    
    # Combine and sort events
    events = peak_idx + valley_idx
    if len(events) == 0:
        events = [0]  # add first day as fallback

    events.sort()
    events_dates = [data.index[event] for event in events]
    return events_dates, data


def get_explain_summary(explanations, stock):
    """
    Generate a summary string from explanations.

    Args:
        explanations (list): List of explanation dicts.

    Returns:
        str: Formatted summary string.
    """
    summary_lines = []
    for idx, ex in enumerate(explanations):
        line = f"""{idx}. From {ex['date']} to {ex['end_date']}, the {stock} stock price went from {f"{ex['price']:.2f}"} to {f"{ex['end_price']:.2f}"} following {ex['trend']} trend by {ex['pct_change']:.2f}% price change. 
{ex['explanation']}
"""
        summary_lines.append(line)
    list_events = "\n".join(summary_lines)
    return list_events

def plot_fig(data, event_dates, stock, outname, movingaverage = 3):
    # Plot price data
    plt.figure(figsize=(14, 6))
    plt.plot(data.index, data['Close'], label='Close Price')

    # Calculate moving average for plotting
    
    data['MovingAverage'] = data['Close'].rolling(window=movingaverage).mean()
    plt.plot(data.index, data['MovingAverage'], label=f'{movingaverage}-Day Moving Average', color='orange')

    # Mark event dates
    for event_date in event_dates:
        plt.axvline(event_date, color='red', linestyle='--', alpha=0.7)

    plt.title(f"{stock} Close Price with Event Dates")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()

    fname = os.path.join(OUTPUT_E, outname)
    plt.savefig(fname)
    return fname
    

# ================ main explain function ===================


async def explain(stock=None, 
            past_date=None, 
            present_date=None, 
            data=None,
            movingaverage:int=3, 
            provider = None,
            sp500=False,
            include_fundamentals=False,
            min_time_diff=10,
            min_price_diff=0.1, 
            max_events=5):
    # Use the unified collect_events function
    if data is not None:
        event_dates, data = collect_events(data=data, movingaverage=movingaverage)
    else:
        event_dates, data = collect_events(stock=stock, past_date=past_date, present_date=present_date, movingaverage=movingaverage)
    
    close_prices = data['Close']

    # Handle cases where event_dates is empty or has only one element
    if not event_dates:
        return [], [], data
    
    event_dates_ = event_dates.copy()
    event_dates_.append(data.index[-1]) # add last date to explain till the end

    # Calculate percentage change for each event
    explanations_temp = []
    for i in range(len(event_dates_)-1):
        start_date = data['Date'].loc[event_dates_[i]]
        end_date = data['Date'].loc[event_dates_[i+1]]
        start_price = close_prices.loc[event_dates_[i]]
        end_price = close_prices.loc[event_dates_[i+1]]
        pct_change = abs((end_price - start_price) / (start_price if start_price != 0 else 1e-8)) * 100
        explanations_temp.append({
            'start_date': start_date,
            'end_date': end_date,
            'start_price': start_price,
            'end_price': end_price,
            'pct_change': pct_change
        })

    # Sort by percentage change and limit to max_events
    top_events = sorted(explanations_temp, key=lambda x: x['pct_change'], reverse=True)[:max_events]
    print(f"Explaining {len(top_events)} significant events for {stock}")
    explanations = []
    event_d = []
    
    for i in range(len(top_events)):
        event = top_events[i]
        start_date = event['start_date']
        end_date = event['end_date']
        start_price = event['start_price']
        end_price = event['end_price']
        pct_change = event['pct_change']
        
        print(f"Explaining price movement from {start_date} to {end_date}")
        direction = "up" if end_price > start_price else "down"
        
        # Generate regular explanation
        if sp500:
            prompt = MACRO_EXPLAIN_PROMPT.format(direction=direction, 
                                        start_date=start_date.strftime(DATE_FORMAT), 
                                        start_price=f"{start_price:.2f}", 
                                        end_date=end_date.strftime(DATE_FORMAT), 
                                        end_price=f"{end_price:.2f}",
                                        percentage=f"{pct_change:.2f}%")
        else:
            prompt = EXPLAIN_PROMPT.format(stock=stock, 
                                        direction=direction, 
                                        start_date=start_date.strftime(DATE_FORMAT), 
                                        start_price=f"{start_price:.2f}", 
                                        end_date=end_date.strftime(DATE_FORMAT), 
                                        end_price=f"{end_price:.2f}",
                                        percentage=f"{pct_change:.2f}%")
      
        answer = await call_ai(prompt, web_search=True, provider=provider)
        
        explanation_entry = {
            'date': start_date.strftime(DATE_FORMAT), 
            'end_date': end_date.strftime(DATE_FORMAT), 
            'explanation': answer, 
            'pct_change': pct_change,
            'price': float(start_price), 
            'end_price': float(end_price), 
            'trend': direction
        }
        
        # Generate fundamental explanation if requested
        if include_fundamentals:
            prompt2 = EXPLAIN_FUNDAMENTAL_PROMPT.format(stock=stock, 
                                        direction=direction, 
                                        start_date=start_date.strftime(DATE_FORMAT), 
                                        start_price=f"{start_price:.2f}", 
                                        end_date=end_date.strftime(DATE_FORMAT), 
                                        end_price=f"{end_price:.2f}",
                                        percentage=f"{pct_change:.2f}%")
            answer2 = await call_ai(prompt2, web_search=True, provider=provider)
            explanation_entry['fundamental_explanation'] = answer2
        
        explanations.append(explanation_entry)
        event_d.append(start_date.strftime(DATE_FORMAT))
    
    return explanations, event_d, data

if __name__ == "__main__": 
    stock = "PLTR"
    past_date = "2020-01-01"
    present_date = "2025-01-16"

    # Load data and collect events using unified function
    events, data = collect_events(
        stock=stock, 
        past_date=past_date, 
        present_date=present_date,
        movingaverage=30, 
        filter_method='topk', 
        topk=7
    )

    plot_fig(data, events, stock, outname=f"{stock}_explain_events7.png", movingaverage=30)
    print("Summary of explanations:")
    print(events)