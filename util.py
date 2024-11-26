import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict

# 데이터 전처리 함수
def process_data(data_path, cat_features, num_features, test_size=0.3, random_state=42):
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
    
    # 수치형 데이터 정규화
    scaler = StandardScaler()
    num_df = pd.DataFrame(scaler.fit_transform(df[num_features]), columns=num_features)
    

    ######### 범주형 속성 처리 ##########
    
    cat_df = df[cat_features].copy()
    # 'Error Message' 컬럼의 NaN 값을 'None'으로 대체
    cat_df['Error Message'] = cat_df['Error Message'].fillna('None')
    label_encoders = {}
    # 범주형 데이터 레이블 인코딩하여 nn.Embedding의 입력으로 사용 가능
    for feature in cat_features:
        le = LabelEncoder()
        cat_df[feature] = le.fit_transform(cat_df[feature].astype(str))
        label_encoders[feature] = le

    y = (df['Is Fraud?'] == 'Yes').astype(int)
    
    # train-validation 분리
    train_cat_X, valid_cat_X, train_y, valid_y = train_test_split(
        cat_df,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # train-validation 분리 (수치형)
    train_num_X, valid_num_X, _, _ = train_test_split(
        num_df,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return (train_cat_X, train_num_X, train_y), (valid_cat_X, valid_num_X, valid_y)

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