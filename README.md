# Atradebot

![image](img/Atradebot.jpg)

Yet another Bot to help you choose what to invest using AI 

[Documentation](https://atradebot.readthedocs.io/en/latest/index.html#)

## Table of Contents

- [API keys](#api-keys)
- [Install](#install)
- [How to run with OpenAI](#how-to-run-with-openai)
- [How to run locally](#how-to-run-locally)
    - [Install LMStudio](#install-lmstudio)
    - [Install websearch](#install-websearch)
    - [Run](#run)
- [DISCLAIMER](#disclaimer)
    - [Specific Disclaimers](#specific-disclaimers)
    - [Limitation of Liability](#limitation-of-liability)
- [License](#license)

## API keys:

Get your api keys from:

[OpenAI](https://openai.com/index/openai-api/) for running the AI agents in the cloud

[Alpaca](https://alpaca.markets/) for doing trades live or paper trade

Create your Alpaca paper trade account by following these [steps](https://alpaca.markets/learn/start-paper-trading) 

Create a .env file for your API keys. You can use `.env.example`

```bash
cp .env.example .env
```

Open and edit the .env file to add your API keys


## Install

Create a `uv` environment:

```bash
# install uv if you dont have it
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv trade --python 3.14

source .trade/bin/activate
```

Then git clone and install from source:

```bash
git clone https://github.com/Andrechang/Atradebot
cd Atradebot
pip install -e .
```


## How to run with OpenAI

Atradebot `main.py` does 3 main steps: 
 - search for potential invesment opportunities
 - rank each stock in current portifolio 
 - manage your portifolio by buying or selling stocks

from the command line, use the following:

```bash
python src/atradebot/main.py --provider openai --mode llm --portfolio_path path/to/your/portfolio.csv
```

- `--provider`: Choose the LLM provider (`openai` or `lmstudio`).
- `--mode`: Set to `llm` for standard LLM mode or `agent` for agent-based mode.
- `--portfolio_path`: (Optional) Path to your portfolio CSV file. If none provided, it will get it from Alpaca account
- `--execute`: (Optional) Add this flag to actually execute trades in alpaca.

This will generate a full .pkl file report into `sd/outputs` folder.

You can convert the .pkl file into pdf using 

```bash
python src/atradebot/utils/plot.py
```



## How to run locally

If you have a GPU with 24GB or more then you can run locally.

### Install LMStudio

Download LMStudio app [here](https://lmstudio.ai/)

Then run: 

`./LM-Studio-0.3.36-1-arm64.AppImage --no-sandbox` 

can also start in cli: 

`lms server start` [ref](https://lmstudio.ai/blog/dgx-spark)
`lms get openai/gpt-oss-20b`


Note: In the GUI, remember to increase context length of loaded model

### Install websearch

For local websearch MCP, install [websearch-mcp](https://github.com/mrkrsl/web-search-mcp)

```bash
git clone https://github.com/mrkrsl/web-search-mcp
cd web-search-mcp
npm install
npx playwright install
npm run build
```

Then run this to start MCP locally

```bash
# Start on default port 8000
npm run start:http
```

When running in HTTP mode, the server provides:
- **Health Check**: `GET http://127.0.0.1:8000/health`

Here is a curl command to test

```bash
curl -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "get-web-search-summaries",
      "arguments": {
        "query": "AI news 2024",
        "limit": 2
      }
    }
  }'
```

### Run

Finally main atradebot

```bash
python src/atradebot/main.py --provider lmstudio --mode llm --portfolio_path path/to/your/portfolio.csv
```








## DISCLAIMER

The financial trading software provided herein is for educational and informational purposes only and does not constitute investment advice. Past performance is not a guarantee of future results. Investing involves risk, including the possible loss of principal. You should always consult with a qualified financial advisor before making any investment decisions.

### Specific Disclaimers

The software is not a recommendation to buy or sell any security or financial instrument.
The software does not take into account your individual investment objectives, financial situation, or needs.
The software is not a substitute for professional financial advice.
You are solely responsible for the risks associated with your investment decisions.


### Limitation of Liability

In no event shall this software or any of his maintaners be liable for any direct, indirect, incidental, special, punitive, or consequential damages arising out of the use of this software.


## License

Atradebot is open-source software released under the [Apache 2.0 license](https://github.com/Superalgos/Superalgos/blob/master/LICENSE)