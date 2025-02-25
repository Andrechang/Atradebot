## Atradebot

![image](img/Atradebot.jpg)

Yet another Bot to help you choose what to invest using AI 

[Documentation](https://atradebot.readthedocs.io/en/latest/index.html#)

### API keys:

Get your api keys from:

[Finhub](https://finnhub.io/) 
[OpenAI](https://openai.com/index/openai-api/)
[Alpaca](https://alpaca.markets/)
[tavily](https://tavily.com/)

```
export FINNHUB_API_KEY=<api key>
export OPENAI_API_KEY=<api key>
export ALPACA_API=<api key>
export ALPACA_SECRET=<api key>
export TAVILY_API_KEY=<api key>
```

And set it to your environment variables


### Install

Requirement:

```
pip install -r requirements.txt
```

Install using pip

```
pip install -e .
```

if errors do:
```
pip install --upgrade pip
```

## Regression test

`pytest -m "test" tests/test_main.py`

## How to run

1. to generate full report and visualize results

download current portifolio .csv file from broker and put in sd/


```
python main.py -s GPTStrategy -n finhub -i 20 -f 30
```

this will generate files in sd/output/
then visualize with:

```
python dashboard.py
```

2. auto trade:
run gen_allocation to place buy/sell in alpaca API

```
python autotrade.py
```

# License

Atradebot is open-source software released under the [Apache 2.0 license](https://github.com/Superalgos/Superalgos/blob/master/LICENSE)