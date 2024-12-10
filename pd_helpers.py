# Charlie Bauer
# 11-17-23
import pandas as pd
import numpy as np
from itertools import product

def set_obj_dtypes(df: pd.DataFrame, obj_cols=[]) -> pd.DataFrame:
    """
    Return:
        df with obj_cols converted from strings to lists or dicts
    """
    df1 = df.copy()
    for col in obj_cols:
        df1[col] = df1[col].apply(lambda obj_str: eval(obj_str))
    return df1

def daily_to_monthly(df: pd.DataFrame, date_col='date', cat_col=[]) -> pd.DataFrame:
    """
    Params:
        df: DataFrame with date_col and cat_col as columns
        date_col: name of the date column in df
        cat_col: list of names of the categorical columns to keep
    Return:
        df with date_col aggregated to monlth-level and grouped by cat_col
    """
    df1 = df.copy()
    df1[date_col] = pd.to_datetime(df1[date_col])
    df1['date'] =  df1[date_col].dt.to_period('M').dt.to_timestamp()
    df1 = df1.drop(columns=date_col).groupby(["date"]+cat_col).sum(numeric_only=True)
    return df1.reset_index()

def rolling_1yr(df: pd.DataFrame, stop_month=None, date_col='date') -> pd.DataFrame:
    """
    Params:
        df: DataFrame with date column
        stop_month (optional): The exclusive stopping month for the rolling_1yr DataFrame.
                For example, if stop_month='2023-11-25' then November 2023 is the stopping month
                and the dates '2022-11-01' through '2023-10-31' will be included.
                If None, then the last month in the dataset will be the stopping month and the returned
                df1 will include values up to and excluding the last month.
        date_col: name of the date column in df
    Return:
        df filtered to the last 12 months of full data
    """
    if stop_month is None:
        stop_month = df[date_col].max()
    stop_month = pd.to_datetime(pd.to_datetime(stop_month).strftime('%Y-%m-01'))
    df1 = df.copy()
    df1 = df1[(df1[date_col] >= stop_month - pd.DateOffset(years=1)) & (df1[date_col] < stop_month)]
    return df1.sort_values(date_col).reset_index(drop=True)

def full_col(df: pd.DataFrame, col: str, vals) -> pd.DataFrame:
    """
    Params:
        df: DataFrame with col as column
        col: name of column
        vals: list of values that col should have
    Return:
        df with new rows added so that all vals are present in col.
        New rows have zeros for other columns.
    """
    df1 = df.reset_index(drop=True)
    present = sorted(list(pd.unique(df[col])))
    for val in vals:
        if val not in present:
            df1.loc[len(df1)] = {c: val if c == col
                                 else pd.to_datetime(0) if c == 'date' 
                                 else "0" 
                                 for c in df.columns}
    return df1

def full_df(df: pd.DataFrame, cat_col=[]) -> pd.DataFrame:
    """
    Params:
        df: DataFrame with cat_col as columns
        cat_col: list of names of all categorical columns
    Return:
        df with new rows added so that all combinations of values
        for cat_col are present. New rows have zeros for other columns.
    """
    vals = [[val for val in np.sort(pd.unique(df[c])) if val != 0] for c in cat_col]
    all_cat = list(product(*vals))
    df1 = pd.DataFrame(all_cat, columns=cat_col)
    return pd.merge(df1, df, on=cat_col, how='left').fillna(0)