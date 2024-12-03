import pandas as pd
import torch
from jupyterlab.semver import valid
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List

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
def translation(df: pd.DataFrame) -> pd.DataFrame :
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

    trans = zipcode(trans)

    trans = convert_date_to_weekday(trans, 'Year', 'Month', 'Day')

    # whole dataset
    return trans

# MUST BE EXECUTED AFTER TRANSITION
def discard(df: pd.DataFrame, discarded : List[str]) -> pd.DataFrame :
    if not all(feature in df.columns for feature in discarded) :
        raise ValueError("DISCARD ValueError : NO COLUMN IN GIVEN DATASET")

    # whole dataset
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

def zipcode(df) :
    trans = df.copy()
    trans['Zipcode'] = (trans['Zipcode'] // 10000).astype(str)
    return trans

def convert_date_to_weekday(df, col_Year, col_Month, col_Day):

    # 날짜 컬럼 생성
    df['WeekDay'] = pd.to_datetime(
        df[col_Year].astype(str) + '-' +
        df[col_Month].astype(str).str.zfill(2) + '-' +
        df[col_Day].astype(str).str.zfill(2)
    )

    # 요일 추출 (0=월요일, 6=일요일)
    df['WeekDay'] = df['WeekDay'].dt.dayofweek

    # 요일 이름으로 변환
    weekday_map = {
        0: 'MON',
        1: 'TUE',
        2: 'WED',
        3: 'THU',
        4: 'FRI',
        5: 'SAT',
        6: 'SUN'
    }

    df['WeekDay'] = df['WeekDay'].map(weekday_map)

    return df

## num_features
def iqr(df: pd.DataFrame, num_features: List[str]) -> pd.DataFrame :
    # ValueError - NO COLUMN
    if not all(feature in df.columns for feature in num_features) :
        raise ValueError("IQR ValueError : NO COLUMN IN GIVEN DATASET")

    num_df = df.copy()
    mask = pd.Series(True, index=df.index)

    for col in num_features :
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        l_bound = Q1 - IQR_MULTIPLIER * IQR
        u_bound = Q3 + IQR_MULTIPLIER * IQR
        num_df = num_df[(num_df[col] >= l_bound) & (num_df[col] <= u_bound)]

    return num_df

def scale(train_df: pd.DataFrame, valid_df: pd.DataFrame, num_features: List[str]) -> Tuple[pd.DataFrame , pd.DataFrame] :
    # ValueError - NO COLUMN
    if not all(feature in train_df.columns for feature in num_features) :
        raise ValueError("SCALE ValueError : NO COLUMN IN GIVEN DATASET")
    if not all(feature in valid_df.columns for feature in num_features) :
        raise ValueError("SCALE ValueError : NO COLUMN IN GIVEN DATASET")

    scaler = StandardScaler()
    scaler.fit(train_df[num_features])

    train_df[num_features] = pd.DataFrame(
        scaler.transform(train_df[num_features]),
        columns = num_features,
        index = train_df.index,
    )
    valid_df[num_features] = pd.DataFrame(
        scaler.transform(valid_df[num_features]),
        columns = num_features,
        index = valid_df.index,
    )

    return train_df, valid_df

## cat_features
def encode(train_df: pd.DataFrame, valid_df: pd.DataFrame, cat_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, any] :
    # ValueError - NO COLUMN
    if not all(feature in train_df.columns for feature in cat_features) :
        raise ValueError("ENCODE ValueError : NO COLUMN IN GIVEN DATASET")
    if not all(feature in valid_df.columns for feature in cat_features) :
        raise ValueError("ENCODE ValueError : NO COLUMN IN GIVEN DATASET")

    label_encoders = {}

    for col in cat_features :
        le = LabelEncoder()
        le.fit(train_df[col].astype(str))
        train_df[col] = le.transform(train_df[col].astype(str))
        valid_df[col] = le.transform(valid_df[col].astype(str))
        label_encoders[col] = le

    return train_df, valid_df, label_encoders

def get_target(df: pd.DataFrame) -> pd.DataFrame :
    if 'Is Fraud?' not in df.columns:
        raise ValueError("GET_TARGET ValueError : NO TARGET IN GIVEN DATASET")

    target = pd.DataFrame(df['Is Fraud?'] == 'Yes').astype(float)

    return target

def discard_label(df: pd.DataFrame) -> pd.DataFrame :
    if 'Is Fraud?' not in df.columns:
        raise ValueError("DISCARD_TARGET ValueError : NO TARGET IN GIVEN DATASET")

    unlabeled = df[df['Is Fraud?'] == "No"]

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
    df = translation(df)

    print("IQR")
    df = iqr(df, num_features)

    print("SPLIT")
    train_df, valid_df = split_by_date(df)

    print("DISCARD")
    train_df = discard(train_df, discarded)
    valid_df = discard(valid_df, discarded)

    print("SCALE")
    train_df, valid_df = scale(train_df, valid_df, num_features)

    print("ENCODE")
    train_df, valid_df, label_encoders = encode(train_df, valid_df, cat_features)

    print("UNLABEL")
    train_df = discard_label(train_df)

    print("TARGET")
    train_y = get_target(train_df)
    valid_y = get_target(valid_df)

    print("TRAIN CAT/NUM")
    train_cat_x = train_df[cat_features]
    train_num_x = train_df[num_features]

    print("VALID CAT/NUM")
    valid_num_x = valid_df[num_features]
    valid_cat_x = valid_df[cat_features]

    print("RETURN")
    return (train_cat_x, train_num_x, train_y), (valid_cat_x, valid_num_x, valid_y), label_encoders

def dt_process_data(data_path: str, cat_features: List[str], num_features: List[str], discarded: List[str]) -> Tuple:
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
    df = translation(df)

    print("IQR")
    df = iqr(df, num_features)

    print("SPLIT")
    train_df, valid_df = split_by_date(df)

    print("DISCARD")
    train_df = discard(train_df, discarded)
    valid_df = discard(valid_df, discarded)

    print("SCALE")
    train_df, valid_df = scale(train_df, valid_df, num_features)

    print("ENCODE")
    train_df, valid_df, label_encoders = encode(train_df, valid_df, cat_features)

    print("TARGET")
    train_y = get_target(train_df)
    valid_y = get_target(valid_df)

    print("TRAIN CAT/NUM")
    train_cat_x = train_df[cat_features]
    train_num_x = train_df[num_features]

    print("VALID CAT/NUM")
    valid_num_x = valid_df[num_features]
    valid_cat_x = valid_df[cat_features]

    print("RETURN")
    return (train_cat_x, train_num_x, train_y), (valid_cat_x, valid_num_x, valid_y), label_encoders

