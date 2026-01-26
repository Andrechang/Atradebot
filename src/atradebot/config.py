# a common place to add file paths and parameters here

# project global defines
import os
import json
from dataclasses import dataclass
import diskcache as dc
import logging
from dotenv import load_dotenv
import git

def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root

DATE_FORMAT = "%Y-%m-%d" # same as yfinance

# autotrade configs
FUTURE = 'future_price_in_6_months' # or 'future_price_in_6_months' or 'future_price_in_1_year'
FILTER = False  # filter out large positions stocks 
PAPERTRADE = True # use paper trading account alpaca


# folders
ROOT_D = get_git_root(os.path.abspath(__file__)) + '/'

FILES_D = os.path.join(ROOT_D, 'sd')

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set log level
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(FILES_D, 'app.log')),  # Log to file
        logging.StreamHandler()  # Print to terminal
    ]
)


TMP_FILE = os.path.join(FILES_D, 'tmp.csv')
OUTPUT_YFDATA = os.path.join(FILES_D, 'data') # directory for yfinance cache files (stockstash)

OUTPUT_D = os.path.join(FILES_D, 'output') # directory for search output files
if not os.path.exists(OUTPUT_D):
    os.mkdir(OUTPUT_D)

OUTPUT_E = os.path.join(FILES_D, 'explanations') # directory for explanation output files
if not os.path.exists(OUTPUT_E):
    os.mkdir(OUTPUT_E)

NEWSCACHE_DIR = os.path.join(FILES_D, 'news_cache')
NEWSCACHE = dc.Cache(NEWSCACHE_DIR) # cache news data for inference 
FHUB_API = os.environ.get('FINNHUB_API_KEY') # for getting news finhub


# Data collection
# load_dotenv(os.path.join(FILES_D, 'gptr.env')) # gpt-research
load_dotenv()

HF_LOCALDATA = f'{FILES_D}/atradebot_collect_' # collect to add to dataset (cronjob) collect.py
HF_DATASET = 'atradebot_dataset' # dataset for training
HF_STOCKDB = f'{OUTPUT_E}/atradebot_stockdb.jsonl' # collect to add to dataset (collect_explain.py) for rag_system


# List of tickers
# get all stock tickers from https://www.sec.gov/files/company_tickers.json
TICKERS = os.path.join(FILES_D, 'company_tickers.json') # list of stock tickers
with open(TICKERS, 'r') as f:
    alltickers = json.load(f)
ALLTICKS = [v['ticker'] for _, v in alltickers.items()] + ['BYDDY']
rmticks = ['AI', 'IT', 'AR', 'VR', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
ALLTICKS = [t for t in ALLTICKS if t not in rmticks]


#APIs
try:
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY') # for LLM
    ALPACA_API = os.environ.get('ALPACA_API_KEY') # trading platform - US market
    ALPACA_SECRET = os.environ.get('ALPACA_SECRET_KEY')
except Exception as e:
    print(f"An API KEY error occurred: {e}")


# LLM Provider Options
class LLMProvider:
    # cloud models
    OPENAI = "openai"
    # local models
    LMSTUDIO = "lmstudio"
    # debug
    DEBUG = "debug"

@dataclass
class SearchConfig:
    SECTOR: str = 'information technology'
    AGENT_MODE: bool = False  # use langchain agent for research
    # LLM Provider Configuration
    # Options: "openai", "claude", "gemini", "ollama", "vllm", "qwen", "debug"
    LLM_PROVIDER: str = LLMProvider.LMSTUDIO  # default LLM provider

    # cloud models configurations for each provider
    OPENAI_MODEL: str = "gpt-5-mini" #"gpt-5-mini" #"gpt-5-mini" #"gpt-4"

    # local models
    LMSTUDIO_MODEL: str = "nvidia/nemotron-3-nano"

    LMSTUDIO_BASE_URL: str = "http://192.168.15.9:1234/v1"
    
    # Enable web search capabilities (mainly for Claude)
    ENABLE_WEB_SEARCH: bool = True

CONFIG = SearchConfig()