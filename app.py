import streamlit as st

st.set_page_config(page_title="Atradebot", page_icon="🤖", layout="wide")

st.markdown(
    """
    Click on the pages on the left to navigate through different parts of atradebot.

    ### [Tests](/page1): 
    Test different strategies and models:
    1. SimpleStrategy: Test average cost strategy and sharpe allocation
    2. FinForecastStrategy: Test strategy using model to forecast stock gain/loss based on news sentiment

    [Github](https://github.com/Andrechang/Atradebot)

    [Documentation](https://atradebot.readthedocs.io/en/latest/index.html#)
"""
)