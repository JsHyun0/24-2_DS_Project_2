import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder



def IQR(df, columns) :
    for col in columns :
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        l_bound = Q1 - 1.5 * IQR
        u_bound = Q3 + 1.5 * IQR

        # 극값 제거
        df = df[(df[col] >= l_bound) & (df[col] <= u_bound)]
    return df

def Scale(df, num_features) :
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df[num_features]), columns=num_features)

    return scaled # return num_features

def Cleanse(df, cat_features) :
    cleaned = df[cat_features].copy()

    cleaned['Error Message'] = cleaned['Error Message'].fillna('None')

    label_encoders = {}

    for col in cat_features :
        le = LabelEncoder()
        cleaned[col] = le.fit_transform(cleaned[col].astype(str))
        label_encoders[col] = le

    return cleaned # return cat_features

def Target(df) :
    y = df['Is Fraud?'] == 1
    return y

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

class PreProc :
    def __init__(self) :
        super().__init__()

    def preprocess(self, DataSet, cat_features, num_features, discarded) :
        # discard not-need features
        df = DataSet.copy()

        print("target mapping...")
        # Fraud mapping
        y = (df['Is Fraud?'] == 'Yes').astype(int)

        print("discard features...")
        df = df.drop(columns=discarded)

        print("num_features IQR...")
        # numerical features
        num_df = IQR(df, num_features)

        print("num_features Scaling...")
        num_df = Scale(num_df, num_features)

        print("cat_features Cleansing...")
        cat_df = Cleanse(df, cat_features)

        print("complete preprocessing...")
        return (cat_df, num_df)

    def TVsplit(self, cat_df, num_df, y, test_size=0.3, random_state=42) :

        train_cat_x, valid_cat_x, train_y, valid_y = train_test_split(
            cat_df,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        train_num_x, valid_num_x, _, _ = train_test_split(
            num_df,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        return (train_cat_x, train_num_x, train_y), (valid_cat_x, valid_num_x, valid_y)

