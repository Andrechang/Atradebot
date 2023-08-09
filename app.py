import streamlit as st

st.set_page_config(page_title="Atradebot", page_icon="🤖", layout="wide")

st.markdown(
    """
    Click on the pages on the left to navigate through different parts of atradebot.

    ## page0: app to calculate risk level and recommend stocks based on saved portfolio

    ## page1: app test average cost strategy
    
    ## page2: app test strategy using model to forecast stock gain/loss based on news sentiment

"""
)