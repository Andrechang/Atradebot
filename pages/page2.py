import streamlit as st
from streamlit_extras.switch_page_button import switch_page

"""
# Generating Stock allocation
"""

if 'stocks_choice' not in st.session_state:
    st.session_state.stocks_choice = {}


st.write(st.session_state.qa_answers)


#generate plan

#generate plots if choices done in past

#generate plots for future projection: (highlight news)


submitted = st.button("Learn more")
if submitted:
    switch_page("page3")