import torch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from bokeh.command.subcommands.sampledata import Sampledata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List

from utils.preprocess import split_data_by_time

# Constant Definition
TRAIN_MONTH_START = 1
TRAIN_MONTH_END = 9
IQR_MULTIPLIER = 1.5
NONE_LABEL = 'None'

# need to separate by models
def evaluate_validation(model, data):
    model.eval()
    with torch.no_grad():
        y_pred = model(data)
        return y_pred

def evaluate_test(model, data):
    model.eval()
    with torch.no_grad():
        y_pred = model(data)
        return y_pred


# Data Preprocess
## feature modify
def transition(df: pd.DataFrame) -> pd.DataFrame :
    if 'Acct Open Date' not in df.columns :
        raise ValueError("TRANSITION ValueError : 'Acct Open Date' COLUMN DOES NOT EXIST")

    if 'Error Message' not in df.columns :
        raise ValueError("TRANSITION ValueError : 'Error Message' COLUMN DOES NOT EXIST")

    if 'Year' not in df.columns :
        raise ValueError("Year ValueError : 'Year' COLUMN DOES NOT EXIST")

    if 'Month' not in df.columns :
        raise ValueError("Month ValueError : 'Month' COLUMN DOES NOT EXIST")

    if 'Day' not in df.columns :
        raise ValueError("Day ValueError : 'Day' COLUMN DOES NOT EXIST")

    trans = df.copy()

    trans['Error Message'] = trans['Error Message'].astype(bool)

    trans['Since Open Month'] = (trans['Year'] - trans['Acct Open Date'].str[-4:].astype(int)) * 12 + (trans['Month'] - trans['Acct Open Date'].str[:2].astype(int)).astype(int)

    return trans

# MUST BE EXECUTED AFTER TRANSITION
def discard(df: pd.DataFrame, discarded : List[str]) -> pd.DataFrame :
    if not all(feature in df.columns for feature in discarded) :
        raise ValueError("DISCARD ValueError : NO COLUMN IN GIVEN DATASET")

    return df.drop(columns=discarded)

## Train-Valid Set split
def split_by_date(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame] :
    # Value Error : There is no any Month in given DATASET
    if 'Month' not in df.columns :
        raise ValueError("SPLIT ValueError : 'Month' COLUMN DOES NOT EXIST.")

    mask = df['Month'].between(TRAIN_MONTH_START, TRAIN_MONTH_END)
    train_df = df[mask]
    val_df = df[~mask]

    return train_df, val_df

## num_features
def iqr(df: pd.DataFrame, num_features: List[str]) -> pd.DataFrame :
    # ValueError - NO COLUMN
    if not all(feature in df.columns for feature in num_features) :
        raise ValueError("IQR ValueError : NO COLUMN IN GIVEN DATASET")

    num_df = df[num_features]
    mask = pd.Series(True, index=df.index)

    for col in num_features :
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        l_bound = Q1 - IQR_MULTIPLIER * IQR
        u_bound = Q3 + IQR_MULTIPLIER * IQR
        mask &= ~((num_df[col] < l_bound) | (num_df[col] > u_bound))

    result = df.copy()
    result.loc[~mask, num_features] = None

    return result

def scale(df: pd.DataFrame, num_features: List[str]) -> pd.DataFrame :
    # ValueError - NO COLUMN
    if not all(feature in df.columns for feature in num_features) :
        raise ValueError("SCALE ValueError : NO COLUMN IN GIVEN DATASET")

    scaler = StandardScaler()
    scaler.fit(df[num_features])

    scaled = pd.DataFrame(
        scaler.transform(df[num_features]),
        columns = num_features,
        index = df.index,
    )

    return df

## cat_features
def encode(df: pd.DataFrame, cat_features: List[str]) -> Tuple :
    # ValueError - NO COLUMN
    if not all(feature in df.columns for feature in cat_features) :
        raise ValueError("ENCODE ValueError : NO COLUMN IN GIVEN DATASET")

    cleaned = df[cat_features].copy()

    label_encoders = {}

    for col in cat_features :
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        cleaned[col] = le.fit_transform(cleaned[col].astype(str))
        label_encoders[col] = le

    return cleaned, label_encoders

def get_target(df: pd.DataFrame) -> pd.DataFrame :
    if 'Is Fraud?' not in df.columns:
        raise ValueError("GET_TARGET ValueError : NO TARGET IN GIVEN DATASET")

    target = pd.DataFrame(df['Is Fraud?'] == 'Yes')

    return target

def discard_label(df: pd.DataFrame) -> pd.DataFrame :
    if 'Is Fraud?' not in df.columns:
        raise ValueError("DISCARD_TARGET ValueError : NO TARGET IN GIVEN DATASET")

    unlabeled = pd.DataFrame(df.drop(columns=["Is Fraud?"]))

    return unlabeled

def process_data(data_path: str, cat_features: List[str], num_features: List[str], discarded: List[str]) -> Tuple:
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CAN NOT FIND DATA AT : {data_path}")
    except Exception as e:
        raise Exception(f"ERROR RAISED DURING LOADING DATA: {str(e)}")

    if df.empty:
        raise ValueError("ValueError : EMPTY DATASET")

    if 'Is Fraud?' not in df.columns:
        raise ValueError("ValueError : NO TARGET IN GIVEN DATASET")

    label_encoders = {}

    print("TRANSITION")
    df = transition(df)

    print("IQR")
    df[num_features] = iqr(df, num_features)
    print("SCALE")
    df[num_features] = scale(df, num_features)

    print("ENCODE")
    df[cat_features], label_encoders = encode(df, cat_features)

    print("DISCARD")
    df = discard(df, discarded)

    print("GET_TARGET")
    y = get_target(df)

    print("SPLIT")
    train_df, valid_df = split_by_date(df)

    print("train_df unlabel")
    train_df = train_df[train_df['Is Fraud?'] == "No"]

    print("target")
    train_y = get_target(train_df).astype(int)
    valid_y = get_target(valid_df).astype(int)

    print("train num/cat")
    train_cat_x = train_df[cat_features]
    train_num_x = train_df[num_features]

    print("valid num/cat")
    valid_num_x = valid_df[num_features]
    valid_cat_x = valid_df[cat_features]

    print("return")
    return (train_cat_x, train_num_x, train_y), (valid_cat_x, valid_num_x, valid_y), label_encoders

