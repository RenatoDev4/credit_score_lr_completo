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
    annual_income: float,
    interest_rate: float,
    delay_from_due_date: float,
    changed_credit_limit: float,
    num_credit_inquiries: float,
    credit_mix: float,
    outstanding_debt: float,
    payment_of_min_amount: float,
) -> pd.DataFrame:
    """
    Apply transformations to the given input parameters according to the predefined rules.

    Args:
        annual_income (float): The annual income of the individual.
        interest_rate (float): The interest rate for the individual.
        delay_from_due_date (float): The delay from the due date for the individual.
        changed_credit_limit (float): The changed credit limit for the individual.
        num_credit_inquiries (float): The number of credit inquiries for the individual.
        credit_mix (float): The credit mix for the individual.
        outstanding_debt (float): The outstanding debt for the individual.
        payment_of_min_amount (float): The payment of the minimum amount for the individual.

    Returns:
        pandas.DataFrame: A DataFrame with the transformed values for each input parameter.
    """
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


def pred_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a prediction score based on input features in the DataFrame.

    This function calculates a prediction score using the logistic regression model,
    which is based on the input features provided in the DataFrame. It first computes
    a linear prediction score and then applies the logistic function to obtain the
    final score.

    Args:
        df (pd.DataFrame): A DataFrame containing the input features.
            The DataFrame should have the following columns:
            - 'Intercept': Intercept term of the logistic regression model.
            - 'Annual_Income': The annual income of the individual.
            - 'Interest_Rate': The interest rate for the individual.
            - 'Delay_from_due_date': The delay from the due date for the individual.
            - 'Changed_Credit_Limit': The changed credit limit for the individual.
            - 'Num_Credit_Inquiries': The number of credit inquiries for the individual.
            - 'Credit_Mix': The credit mix for the individual.
            - 'Outstanding_Debt': The outstanding debt for the individual.
            - 'Payment_of_Min_Amount': The payment of the minimum amount for the individual.

    Returns:
        pd.DataFrame: A DataFrame containing the prediction score.
            The DataFrame will have two additional columns:
            - 'pred_lin': The linear prediction score.
            - 'score': The final prediction score obtained by applying the logistic function.
    """
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


def fx_score(score: float) -> int:
    """
    Calculate the FX score based on the given input score according to predefined breakpoints.

    Args:
        score (float): The input score to calculate the FX score.

    Returns:
        int: The calculated FX score based on predefined breakpoints.
            Returns -1 if the input score is greater than the highest predefined breakpoint.

    Note:
        This function uses predefined breakpoints to map an input score to an FX score.
        If the input score is less than or equal to the first breakpoint, it returns the corresponding FX score.
        If the input score falls between two consecutive breakpoints, it returns the FX score associated with the lower breakpoint.
        If the input score is greater than the highest predefined breakpoint, it returns -1.
    """
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
