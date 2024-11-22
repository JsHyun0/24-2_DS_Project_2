import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# 데이터 전처리 함수
def process_data(data_path, test_size=0.3, random_state=42):
    df = pd.read_csv(data_path)
    cat_features = df.select_dtypes(include=['object']).columns.tolist()
    num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # 데이터 전처리
    processed_data = pd.concat([df[cat_features], df[num_features]], axis=1)
    processed_data = processed_data.drop(columns=['User'])
    # train-validation 분리
    train_data, valid_data = train_test_split(
        processed_data,
        test_size=test_size,
        random_state=random_state
    )
    
    return train_data, valid_data, cat_features, num_features

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