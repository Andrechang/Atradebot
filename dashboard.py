# Dashboard to view all the info for a portfolio
import dash
from dash import dcc
from dash import html
from argparse import ArgumentParser
import pandas as pd
pd.options.mode.copy_on_write = True

from atradebot.utils.portfolio import Portfolio 
from atradebot.params import ROOT_D, OUTPUT_D
from atradebot.utils.utils import find_business_day, get_latest_report
from atradebot.strategies.strategy import StrategyConfig
from datetime import date
import os
import re
from datetime import datetime

selected_dates = [30, 120, 360]

def get_data():
    portfolio = f'{ROOT_D}/sd/test_portfolio.csv' 
    args = StrategyConfig(
        inference_days=10,
        future_days=10,
    )
    args.present_date = date.today()
    args.past_date = find_business_day(args.present_date, -args.inference_days)
    portfolio = Portfolio(portfolio, args)
    data = {'Category': portfolio.tickers, 'Values': portfolio.portfolio['Value'].tolist()}
    portfolio_pie = pd.DataFrame(data)

    # get report
    feats = ['Date', 'Stock', 'Predictions', 'FutureDate']

    output_path = f'{OUTPUT_D}/'
    print(output_path)
    reports_path = get_latest_report(output_path)
    find_data_path = next((report for report in reports_path if 'find_data' in report), None)
    find_report_path = next((report for report in reports_path if 'find_report' in report), None)
    
    find_data = pd.read_csv(os.path.join(output_path, find_data_path))
    with open(os.path.join(output_path, find_report_path), 'r') as f:
        find_text = f.read()
    data_tmp = find_data[find_data['FutureDate'].isin(selected_dates)]
    find_top5 = data_tmp.nlargest(5, 'Predictions')[feats]
    find_bottom5 = data_tmp.nsmallest(5, 'Predictions')[feats]

    manage_report_path = next((report for report in reports_path if 'manage_data' in report), None)
    
    manage_data = pd.read_csv(os.path.join(output_path, manage_report_path))
    data_tmp = manage_data[manage_data['FutureDate'].isin(selected_dates)]
    manage_top5 = data_tmp.nlargest(5, 'Predictions')[feats]
    manage_bottom5 = data_tmp.nsmallest(5, 'Predictions')[feats]
    return portfolio_pie, find_data, find_text, find_top5, find_bottom5, manage_data, manage_top5, manage_bottom5

# Call get_data() once and store the results in global variables
portfolio_pie, find_data, find_text, find_top5, find_bottom5, manage_data, manage_top5, manage_bottom5 = get_data()

def get_dashboard():
    # after run main.py
    # 1- read portfolio
    # 2- show findstocks report
    # 3- show manage report

    # Use the global variables instead of calling get_data() again
    global portfolio_pie, find_data, find_text, find_top5, find_bottom5, manage_data, manage_top5, manage_bottom5

    # Initialize the Dash app
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("Dashboard Example"),
        
        dcc.Markdown(f"FINDSTOCK REPORT\n {find_text}"),
        html.H3("FINDSTOCK Plot"),
        dcc.Graph(
            id='bar-chart',
            figure={
                'data': [
                    {'x': find_data[find_data['FutureDate'] == future_date]['Stock'], 
                     'y': find_data[find_data['FutureDate'] == future_date]['Predictions'], 
                     'type': 'bar', 
                     'name': f"{future_date} days"}
                    for future_date in selected_dates
                ],
                'layout': {
                    'title': 'Percent Change forecast each stock over time',
                    'xaxis': {'title': 'Stock'},
                    'yaxis': {'title': 'Percent Change (%)', 'range': [-120, 120]}
                }
            }
        ),
        # add rank top5 stocks in table
        html.H3("Top 5 Stocks"),
        html.Table(
            [
            html.Tr([html.Th(col) for col in find_top5.columns])
            ] + [
            html.Tr([html.Td(find_top5.iloc[i][col]) for col in find_top5.columns]) for i in range(len(find_top5))
            ]
        ),

        html.H3("Bottom 5 Stocks"),
        html.Table(
            [
            html.Tr([html.Th(col) for col in find_bottom5.columns])
            ] + [
            html.Tr([html.Td(find_bottom5.iloc[i][col]) for col in find_bottom5.columns]) for i in range(len(find_bottom5))
            ]
        ),
       
        html.H3("Current portfolio"),
        dcc.Graph(
            id='pie-chart',
            figure={
                'data': [
                    {'labels': portfolio_pie['Category'], 'values': portfolio_pie['Values'], 'type': 'pie'}
                ],
                'layout': {
                    'title': 'Pie Chart'
                }
            }
        ),

        html.H3("Model Suggestion"),
        dcc.Graph(
            id='bar-chart2',
            figure={
                'data': [
                    {'x': manage_data[manage_data['FutureDate'] == future_date]['Stock'], 
                     'y': manage_data[manage_data['FutureDate'] == future_date]['Predictions'], 
                     'type': 'bar', 
                     'name': f"{future_date} days"}
                    for future_date in selected_dates
                ],
                'layout': {
                    'title': 'Percent Change forecast each stock over time',
                    'xaxis': {'title': 'Stock'},
                    'yaxis': {'title': 'Percent Change (%)', 'range': [-120, 120]}
                }
            }
        ),
                # add rank top5 stocks in table
        html.H3("Top 5 Stocks"),
        html.Table(
            [
            html.Tr([html.Th(col) for col in manage_top5.columns])
            ] + [
            html.Tr([html.Td(manage_top5.iloc[i][col]) for col in manage_top5.columns]) for i in range(len(manage_top5))
            ]
        ),

        html.H3("Bottom 5 Stocks"),
        html.Table(
            [
            html.Tr([html.Th(col) for col in manage_bottom5.columns])
            ] + [
            html.Tr([html.Td(manage_bottom5.iloc[i][col]) for col in manage_bottom5.columns]) for i in range(len(manage_bottom5))
            ]
        ),
       

    ])
    return app

# Run the app
if __name__ == '__main__':
    app = get_dashboard()
    app.run_server(debug=True)
