# app to calculate risk level and recommend stocks based on saved portfolio
import streamlit as st


if 'qa_answers' not in st.session_state:
    st.session_state.qa_answers = {}

risk_categories = {
        1: "Lowest Risk",
        2: "Low Risk",
        3: "Moderate-Low Risk",
        4: "Moderate Risk",
        5: "Moderate-High Risk",
        6: "High Risk",
        7: "Highest Risk"
    }

def calc_risk(answers, amount, timeh):
    # Define risk categories and their corresponding risk levels
    weights = [1, 2, 3, 3]
    # Calculate the weighted score (excluding the 6th question)
    weighted_score = sum([int(answers[i]) * weights[i] for i in range(len(answers))]) + (amount * timeh)
    # Determine the risk level based on the weighted score
    if weighted_score <= 10:
        risk_level = 1
    elif weighted_score <= 20:
        risk_level = 2
    elif weighted_score <= 30:
        risk_level = 3
    elif weighted_score <= 40:
        risk_level = 4
    elif weighted_score <= 50:
        risk_level = 5
    elif weighted_score <= 60:
        risk_level = 6
    else:
        risk_level = 7

    return risk_level

st.write("""
# Answer the questions below to help AI generate a personalized set of stocks:
""")

options = [["Under 25", "25-34", "35-44", "45 or older"],
["To generate income in the short-term","To build wealth in the long-term", "To preserve my capital"],
["I would sell immediately", "I would hold onto my investments", "I would buy more"],
["I'm struggling financially and living paycheck to paycheck", "I'm financially stable with a steady income and some savings", "I have a significant amount of savings and a high income","I have a lot of debt and financial obligations. "],
]

question = ["1- What is your age group?",
"2- What is your investment learning goal?",
"3- How would you feel if your investments lost value in the short-term?",
"4- What is your current financial situation?",
]
indexes = []
for iop, option in enumerate(options):
    opt = st.radio(question[iop], option)
    # st.write("index:", option.index(opt))
    idx = option.index(opt)
    indexes.append(idx)

amount = st.text_input('5- How much are you planning to invest?', '0')
if not amount.isnumeric():
    st.write("write a number for question 5")
timeh = st.text_input('6- For how many month are you planning to invest?', '0')
if not timeh.isnumeric():
    st.write("write a number for question 6")

# Every form must have a submit button.
submitted = st.button("Submit")
if submitted and amount.isnumeric() and timeh.isnumeric():
    risk = calc_risk(indexes, int(amount), int(timeh))
    st.session_state.qa_answers = {"risk":risk, "amount":int(amount), "time":int(timeh)}