# project global defines
import os
import diskcache as dc
import json
import logging
import git

def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


DATE_FORMAT = "%Y-%m-%d" # same as yfinance

ROOT_DIR = os.path.abspath(__file__)
ROOT_D = get_git_root(ROOT_DIR) + '/'

OUTPUT_D = f'{ROOT_D}/sd/output/'
if not os.path.exists(OUTPUT_D):
    os.mkdir(OUTPUT_D)

# Data
HF_LOCALDATA = f'{ROOT_D}/sd/atradebot_collect_' # collect to add to dataset (cronjob)
HF_DATASET = 'achang/atradebot_dataset' # dataset for training
NEWSCACHE_DIR = f'{ROOT_D}/sd/news_cache'
NEWSCACHE = dc.Cache(NEWSCACHE_DIR) # cache news data for inference 

YFCACHE = f'{ROOT_D}/sd/' # cache yfinance data for inference
DATA_KEYS = ['Date', 'News', 'Tickers'] #must have keys in dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set log level
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{ROOT_D}/sd/app.log"),  # Log to file
        logging.StreamHandler()  # Print to terminal
    ]
)
# List of tickers
# get all stock tickers from https://www.sec.gov/files/company_tickers.json
TICKERS = f'{ROOT_D}/sd/company_tickers.json' # list of stock tickers
with open(TICKERS, 'r') as f:
    alltickers = json.load(f)
ALLTICKS = [v['ticker'] for _, v in alltickers.items()]
rmticks = ['AI', 'IT', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AR', 'VR']
ALLTICKS = [t for t in ALLTICKS if t not in rmticks]

#APIs
FHUB_API = os.environ.get('FINNHUB_API_KEY') # for getting news finhub
NEWS_API = os.environ.get('NEWS_API_KEY')
OPENAI_API = os.environ.get('OPENAI_API_KEY') # for LLM

ALPACA_API = os.environ.get('ALPACA_API_KEY') # trading platform
ALPACA_SECRET = os.environ.get('ALPACA_SECRET_KEY') 
PAPERTRADE = True # paper trading or live trading