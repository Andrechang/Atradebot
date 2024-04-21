
# Dashboard to view all the info for a portifolio

import dash
from dash import dcc
from dash import html
from argparse import ArgumentParser

import pandas as pd
pd.options.mode.copy_on_write = True



import ta 


# TODO:
    # News of top positions - with sentiment
    # News of trending stocks - with sentiment

    # Pie chart of stocks/assets[x]
    # Pie chart= if it is ETF then break into stocks
    # Pie chart of sectors

    # top 3 growing stocks suggested - model

def get_dashboard():
    
    # get portfolio
    portifolio = ta.Portfolio('FILES/positions.csv')
    beta, alpha = portifolio.calc_beta()
    #search for etfs:
    # etfs = stocks_lst[stocks_lst['Ticker'].str.contains('-')]
    # sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    # stocks_sp500 = sp500.Symbol.to_list()
    # stocks_not_sp500 = stocks_lst[~stocks_lst['Ticker'].isin(stocks_sp500)] # remove stocks that are in the S&P 500

    # get news

    # Create dashboard
    data = {'Category': portifolio.tickers,
            'Values': portifolio.portifolio['Value'].tolist()}
    df = pd.DataFrame(data)

    # Initialize the Dash app
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("Dashboard Example"),
        dcc.Graph(
            id='pie-chart',
            figure={
                'data': [
                    {'labels': df['Category'], 'values': df['Values'], 'type': 'pie'}
                ],
                'layout': {
                    'title': 'Pie Chart'
                }
            }
        ),
        dcc.Markdown('''
            ## Stats:
            Beta: {beta}
            Alpha: {alpha}
        '''.format(beta=beta, alpha=alpha)),
        dcc.Markdown('''
            ## NEWS:
        ''')
    ])
    return app

def get_arg(raw_args=None):
    parser = ArgumentParser(description="parameters")
    parser.add_argument('-m', '--mode', type=str,
                        default='', help='Create dataset for task modes: forecast | allocation ')
    parser.add_argument('-t', '--thub', type=str,
                        default='', help='push to hub folder name for task dataset')
    parser.add_argument('-r', '--rhub', type=str,
                        default='', help='push to hub folder name for raw news data')
    parser.add_argument('--start_date', type=str, default="2022-08-31", help='start date for trading analysis')
    parser.add_argument('--end_date', type=str, default="2023-06-20", help='end data for trading analysis')
    parser.add_argument('--stocks', type=str, default="AAPL AMZN MSFT NVDA TSLA", help='stocks to analize')
    parser.add_argument('--datasrc', type=str, default="finhub", help='data src finhub or google')

    args = parser.parse_args(raw_args)
    return args

# Run the app
if __name__ == '__main__':
    args = get_arg()
    app = get_dashboard()
    app.run_server(debug=True)
