import streamlit as st

from src.models.predict_model import (apply_transformations, fx_score,
                                      pred_score)

st.set_page_config(
    page_title="Credit Score Model - Welcome",
    page_icon="💳",
)

name = "Renato Moraes"
title = "Machine Learning 🤖 | Data Scientist 🧑‍🔬"
linkedin_url = "https://linkedin.com/in/renato-moraes-11b546272"
github_url = "https://github.com/RenatoDev4"
dataset_link = "https://www.kaggle.com/datasets/parisrohan/credit-score-classification"

st.sidebar.title("Credit Card Score Model 💳")
st.sidebar.markdown(f"**About Author:** {name}")
st.sidebar.markdown(f"**Specialization areas:** {title}")
st.sidebar.markdown(f"**Professional Connections:**")
st.sidebar.markdown(f"**[LinkedIn]({linkedin_url})** | **[GitHub]({github_url})**")

st.sidebar.divider()

st.sidebar.markdown("**About the model❗**")
st.sidebar.write(
    "This model calculate the chance of a customer being or not being in default using Logistic Regression,  Weights of Evidence (WoE), Information Values (IV) and finally Variance Inflation Factor (VIF)"
)

st.sidebar.divider()

st.sidebar.markdown("**Used dataset ✅**")
st.sidebar.write(f"**[Kaggle]( {dataset_link} )**   ")

st.sidebar.divider()

st.sidebar.markdown("**Model metrics** 💹")
st.sidebar.write("KS1: **0.024**")
st.sidebar.write("KS2: **0.363**")


st.title("Credit Score Model - Credit Card")
st.markdown("""
This is a DEMO application of a machine learning model to assess the probability of a customer defaulting on a credit card application. Simply fill in the fields below with your customer details and click 'Evaluate'.
""")

annual_income = st.number_input(
    "Client's annual salary", min_value=0, help="Enter client annual salary, Ex: 50000"
)
interest_rate = st.number_input(
    "Interest rate (In %)",
    min_value=0,
    help="Average credit card interest rate in percentage",
)
delay_from_due_date = st.number_input(
    "Delay in due date",
    min_value=0,
    help="Does the customer have any late invoices? Specify how many days the customer's invoice is overdue.",
)
changed_credit_limit = st.number_input(
    "Changed credit limit",
    min_value=0,
    help="Has the customer ever had their limit changed? How many?",
)
num_credit_inquiries = st.number_input("Number of credit inquiries", min_value=0,
                                       help="How many credit inquiries has the customer had in the last few months?")
credit_mix = st.number_input(
    "Credit Mix", min_value=0, max_value=1, help='1 = GOOD, 0 = BAD')
outstanding_debt = st.number_input("Outstanding Debit", min_value=0,
                                   help="The total as well as interest amount of a debt that has yet to be paid")
payment_of_min_amount = st.number_input(
    "Payment of min amount", min_value=0, max_value=1, help="Does the customer pay the minimum of their invoice? YES=1, NO=0")

if st.button("Evaluate"):
    df_copy = apply_transformations(
        annual_income,
        interest_rate,
        delay_from_due_date,
        changed_credit_limit,
        num_credit_inquiries,
        credit_mix,
        outstanding_debt,
        payment_of_min_amount,
    )
    df_copy = pred_score(df_copy)

    df_copy["fx_score"] = df_copy["score"].map(fx_score)

    chance_of_default = abs(df_copy["score"].values[0] * 100 - 100)
    if chance_of_default < 45:
        st.success(
            f'**The chance that this customer will default is:** {chance_of_default:.2f}% / {df_copy["fx_score"].values[0]}'
        )
    elif chance_of_default >= 45 and chance_of_default <= 60:
        st.warning(
            f'**The chance that this customer will default is:** {chance_of_default:.2f}% / {df_copy["fx_score"].values[0]}'
        )
    else:
        st.error(
            f'**The chance that this customer will default is:** {chance_of_default:.2f}% / {df_copy["fx_score"].values[0]}'
        )
