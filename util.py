import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict

# 데이터 전처리 함수
def process_data(data_path, cat_features, num_features):
    df = pd.read_csv(data_path)
    
    ######### 수치형 속성 처리 ##########
    # 이상치 제거 (IQR 방법)
    num_df = df[num_features]
    mask = pd.Series(True, index=df.index)  # 모든 행을 True로 초기화
    
    for feature in num_features:
        Q1 = num_df[feature].quantile(0.25)
        Q3 = num_df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 마스크 업데이트
        mask &= ~((num_df[feature] < lower_bound) | (num_df[feature] > upper_bound))
    
    # 마스크를 사용하여 한 번에 필터링
    df = df[mask]
    
    ######### 시간 기반 train-test 분할 ##########
    train_mask = df['Month'].between(1, 9)
    train_df = df[train_mask]
    valid_df = df[~train_mask]
    
    ######### 수치형 속성 처리 ##########
    # train 데이터로만 fit
    scaler = StandardScaler()
    scaler.fit(train_df[num_features])
    
    # train과 valid 각각 transform
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
    
    ######### 범주형 속성 처리 ##########
    train_cat_df = train_df[cat_features].copy()
    valid_cat_df = valid_df[cat_features].copy()
    
    train_cat_df['Error Message'] = train_cat_df['Error Message'].fillna('None')
    valid_cat_df['Error Message'] = valid_cat_df['Error Message'].fillna('None')
    
    label_encoders = {}
    # train 데이터로 fit하고 train/valid에 각각 transform
    for feature in cat_features:
        le = LabelEncoder()
        le.fit(train_cat_df[feature].astype(str))
        train_cat_df[feature] = le.transform(train_cat_df[feature].astype(str))
        valid_cat_df[feature] = le.transform(valid_cat_df[feature].astype(str))
        label_encoders[feature] = le
    
    # 타겟 변수 처리
    train_y = (train_df['Is Fraud?'] == 'Yes').astype(int)
    valid_y = (valid_df['Is Fraud?'] == 'Yes').astype(int)
    
    return (train_cat_df, train_num_X, train_y), (valid_cat_df, valid_num_X, valid_y)

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