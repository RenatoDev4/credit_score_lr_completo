import numpy as np
import pandas as pd

from src.features.config import (ANNUAL_INCOME_0, ANNUAL_INCOME_1,
                                 ANNUAL_INCOME_2, BROKEN_DEVELOPED,
                                 CHANGED_CREDIT_LIMIT_0,
                                 CHANGED_CREDIT_LIMIT_1,
                                 CHANGED_CREDIT_LIMIT_2, CREDIT_MIX_0,
                                 CREDIT_MIX_1, DELAY_FROM_DUE_DATE_0,
                                 DELAY_FROM_DUE_DATE_1, DELAY_FROM_DUE_DATE_2,
                                 INTERCEPT, INTEREST_RATE_0, INTEREST_RATE_1,
                                 INTEREST_RATE_2, NUM_CREDIT_INQUIRIES_0,
                                 NUM_CREDIT_INQUIRIES_1,
                                 NUM_CREDIT_INQUIRIES_2, OUTSTANDING_DEBT_0,
                                 OUTSTANDING_DEBT_1, OUTSTANDING_DEBT_2,
                                 PAYMENT_OF_MIN_AMOUNT_0,
                                 PAYMENT_OF_MIN_AMOUNT_1, SCORE)


def apply_transformations(
    annual_income,
    interest_rate,
    delay_from_due_date,
    changed_credit_limit,
    num_credit_inquiries,
    credit_mix,
    outstanding_debt,
    payment_of_min_amount,
):
    rules = {
        "Annual_Income": {
            (annual_income < ANNUAL_INCOME_0[0]): ANNUAL_INCOME_0[1],
            (annual_income >= ANNUAL_INCOME_1[0])
            & (annual_income < ANNUAL_INCOME_1[1]): ANNUAL_INCOME_1[2],
            (annual_income >= ANNUAL_INCOME_2[0]): ANNUAL_INCOME_2[1],
        },
        "Interest_Rate": {
            (interest_rate < INTEREST_RATE_0[0]): INTEREST_RATE_0[1],
            (interest_rate >= INTEREST_RATE_1[0])
            & (interest_rate < INTEREST_RATE_1[1]): INTEREST_RATE_1[2],
            (interest_rate >= INTEREST_RATE_2[0]): INTEREST_RATE_2[1],
        },
        "Delay_from_due_date": {
            (delay_from_due_date < DELAY_FROM_DUE_DATE_0[0]): DELAY_FROM_DUE_DATE_0[1],
            (delay_from_due_date >= DELAY_FROM_DUE_DATE_1[0])
            & (delay_from_due_date < DELAY_FROM_DUE_DATE_1[1]): DELAY_FROM_DUE_DATE_1[
                2
            ],
            (delay_from_due_date >= DELAY_FROM_DUE_DATE_2[0]): DELAY_FROM_DUE_DATE_2[1],
        },
        "Changed_Credit_Limit": {
            (changed_credit_limit < CHANGED_CREDIT_LIMIT_0[0]): CHANGED_CREDIT_LIMIT_0[
                1
            ],
            (changed_credit_limit >= CHANGED_CREDIT_LIMIT_0[0])
            & (
                changed_credit_limit < CHANGED_CREDIT_LIMIT_1[1]
            ): CHANGED_CREDIT_LIMIT_1[2],
            (changed_credit_limit >= CHANGED_CREDIT_LIMIT_2[0]): CHANGED_CREDIT_LIMIT_2[
                1
            ],
        },
        "Num_Credit_Inquiries": {
            (num_credit_inquiries < NUM_CREDIT_INQUIRIES_0[0]): NUM_CREDIT_INQUIRIES_0[
                1
            ],
            (num_credit_inquiries >= NUM_CREDIT_INQUIRIES_1[0])
            & (
                num_credit_inquiries < NUM_CREDIT_INQUIRIES_1[1]
            ): NUM_CREDIT_INQUIRIES_1[2],
            (num_credit_inquiries >= NUM_CREDIT_INQUIRIES_2[0]): NUM_CREDIT_INQUIRIES_2[
                1
            ],
        },
        "Credit_Mix": {
            (credit_mix < CREDIT_MIX_0[0]): CREDIT_MIX_0[1],
            (credit_mix >= CREDIT_MIX_1[0]): CREDIT_MIX_1[1],
        },
        "Outstanding_Debt": {
            (outstanding_debt < OUTSTANDING_DEBT_0[0]): OUTSTANDING_DEBT_0[1],
            (outstanding_debt >= OUTSTANDING_DEBT_1[0])
            & (outstanding_debt < OUTSTANDING_DEBT_1[1]): OUTSTANDING_DEBT_1[2],
            (outstanding_debt >= OUTSTANDING_DEBT_2[0]): OUTSTANDING_DEBT_2[1],
        },
        "Payment_of_Min_Amount": {
            (
                payment_of_min_amount < PAYMENT_OF_MIN_AMOUNT_0[0]
            ): PAYMENT_OF_MIN_AMOUNT_0[1],
            (
                payment_of_min_amount >= PAYMENT_OF_MIN_AMOUNT_1[0]
            ): PAYMENT_OF_MIN_AMOUNT_1[1],
        },
    }

    df = pd.DataFrame(
        {
            "Annual_Income": [annual_income],
            "Interest_Rate": [interest_rate],
            "Delay_from_due_date": [delay_from_due_date],
            "Changed_Credit_Limit": [changed_credit_limit],
            "Num_Credit_Inquiries": [num_credit_inquiries],
            "Credit_Mix": [credit_mix],
            "Outstanding_Debt": [outstanding_debt],
            "Payment_of_Min_Amount": [payment_of_min_amount],
        }
    )

    df_copy = df.copy()

    for column, conditions in rules.items():
        for condition, value in conditions.items():
            df_copy[column] = np.where(condition, value, df_copy[column])

    df_copy["Intercept"] = INTERCEPT

    return df_copy


def pred_score(df):
    df["pred_lin"] = (
        df["Intercept"]
        + df["Annual_Income"]
        + df["Interest_Rate"]
        + df["Delay_from_due_date"]
        + df["Changed_Credit_Limit"]
        + df["Num_Credit_Inquiries"]
        + df["Credit_Mix"]
        + df["Outstanding_Debt"]
        + df["Payment_of_Min_Amount"]
    )
    df["score"] = 1 / (1 + np.exp(-df["pred_lin"]))

    return df


def fx_score(score):
    broken_developed = BROKEN_DEVELOPED

    if score <= broken_developed[0]:
        return SCORE[0]
    if score <= broken_developed[1]:
        return SCORE[1]
    if score <= broken_developed[2]:
        return SCORE[2]
    if score <= broken_developed[3]:
        return SCORE[3]
    if score <= broken_developed[4]:
        return SCORE[4]
    return -1
