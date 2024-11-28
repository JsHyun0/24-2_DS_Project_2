import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict

def handle_numerical_features(df, num_features):
    num_df = df[num_features]
    mask = pd.Series(True, index=df.index)
    
    for feature in num_features:
        Q1 = num_df[feature].quantile(0.25)
        Q3 = num_df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask &= ~((num_df[feature] < lower_bound) | (num_df[feature] > upper_bound))
    
    return df[mask]

def split_data_by_time(df):
    train_mask = df['Month'].between(1, 9)
    train_df = df[train_mask]
    valid_df = df[~train_mask]
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
    
    train_cat_df['Error Message'] = train_cat_df['Error Message'].fillna('None')
    valid_cat_df['Error Message'] = valid_cat_df['Error Message'].fillna('None')
    
    label_encoders = {}
    
    for feature in cat_features:
        le = LabelEncoder()
        le.fit(train_cat_df[feature].astype(str))
        train_cat_df[feature] = le.transform(train_cat_df[feature].astype(str))
        valid_cat_df[feature] = le.transform(valid_cat_df[feature].astype(str))
        label_encoders[feature] = le
    
    return train_cat_df, valid_cat_df, label_encoders

def process_data(data_path, cat_features, num_features):
    df = pd.read_csv(data_path)
    
    df = handle_numerical_features(df, num_features)
    train_df, valid_df = split_data_by_time(df)
    train_num_X, valid_num_X = scale_numerical_features(train_df, valid_df, num_features)
    train_cat_df, valid_cat_df, label_encoders = encode_categorical_features(train_df, valid_df, cat_features)
    
    train_y = (train_df['Is Fraud?'] == 'Yes').astype(int)
    valid_y = (valid_df['Is Fraud?'] == 'Yes').astype(int)
    
    return (train_cat_df, train_num_X, train_y), (valid_cat_df, valid_num_X, valid_y)
