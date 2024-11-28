from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict

# 상수 정의
TRAIN_MONTH_START = 1
TRAIN_MONTH_END = 9
IQR_MULTIPLIER = 1.5
NONE_LABEL = 'None'

def handle_numerical_features(df: pd.DataFrame, num_features: List[str]) -> pd.DataFrame:
    """이상치를 제거하여 수치형 특성을 처리합니다."""
    if not all(feature in df.columns for feature in num_features):
        raise ValueError("일부 수치형 특성이 데이터프레임에 존재하지 않습니다.")
    
    num_df = df[num_features]
    mask = pd.Series(True, index=df.index)
    
    for feature in num_features:
        Q1 = num_df[feature].quantile(0.25)
        Q3 = num_df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - IQR_MULTIPLIER * IQR
        upper_bound = Q3 + IQR_MULTIPLIER * IQR
        mask &= ~((num_df[feature] < lower_bound) | (num_df[feature] > upper_bound))
    
    return df[mask]

def split_data_by_time(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """시간을 기준으로 데이터를 훈련셋과 검증셋으로 분할합니다."""
    if 'Month' not in df.columns:
        raise ValueError("'Month' 컬럼이 데이터프레임에 존재하지 않습니다.")
        
    train_mask = df['Month'].between(TRAIN_MONTH_START, TRAIN_MONTH_END)
    train_df = df[train_mask]
    valid_df = df[~train_mask]
    
    if len(train_df) == 0 or len(valid_df) == 0:
        raise ValueError("분할 후 훈련셋 또는 검증셋이 비어있습니다.")
        
    return train_df, valid_df

def scale_numerical_features(train_df, valid_df, num_features):
    scaler = StandardScaler()
    scaler.fit(train_df[num_features])
    
    train_num_X = pd.DataFrame(
        scaler.transform(train_df[num_features]), 
        columns=num_features, 
        index=train_df.index
    )
    valid_num_X = pd.DataFrame(
        scaler.transform(valid_df[num_features]), 
        columns=num_features, 
        index=valid_df.index
    )
    
    return train_num_X, valid_num_X

def encode_categorical_features(train_df, valid_df, cat_features):
    train_cat_df = train_df[cat_features].copy()
    valid_cat_df = valid_df[cat_features].copy()
    # if 'Error Message' in cat_features:
    #     train_cat_df['Error Message'] = train_cat_df['Error Message'].fillna('None')
    #     valid_cat_df['Error Message'] = valid_cat_df['Error Message'].fillna('None')
    
    label_encoders = {}
    
    for feature in cat_features:
        le = LabelEncoder()
        le.fit(train_cat_df[feature].astype(str))
        train_cat_df[feature] = le.transform(train_cat_df[feature].astype(str))
        valid_cat_df[feature] = le.transform(valid_cat_df[feature].astype(str))
        label_encoders[feature] = le
    
    return train_cat_df, valid_cat_df, label_encoders

def process_data(data_path: str, cat_features: List[str], num_features: List[str]) -> Tuple:
    """데이터를 전처리하고 훈련/검증 세트를 반환합니다."""
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
    except Exception as e:
        raise Exception(f"데이터 로딩 중 오류 발생: {str(e)}")
    
    if df.empty:
        raise ValueError("데이터프레임이 비어있습니다.")
        
    if 'Is Fraud?' not in df.columns:
        raise ValueError("목표 변수 'Is Fraud?'가 데이터프레임에 존재하지 않습니다.")
    
    df = handle_numerical_features(df, num_features)
    train_df, valid_df = split_data_by_time(df)
    train_num_X, valid_num_X = scale_numerical_features(train_df, valid_df, num_features)
    train_cat_df, valid_cat_df, label_encoders = encode_categorical_features(train_df, valid_df, cat_features)
    
    train_y = (train_df['Is Fraud?'] == 'Yes').astype(int)
    valid_y = (valid_df['Is Fraud?'] == 'Yes').astype(int)
    
    return (train_cat_df, train_num_X, train_y), (valid_cat_df, valid_num_X, valid_y)
