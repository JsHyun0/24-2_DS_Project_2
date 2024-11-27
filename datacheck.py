import pandas as pd
import torch

data_path = './Data/[24-2 DS_Project2] Data.csv'
df = pd.read_csv(data_path)

print(df.columns)

df_date = df[['Year', 'Month', 'Day']]
print(df.head())

def make_time_series(df):
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df = df.drop(columns=['Year', 'Month', 'Day'])
    df = df.set_index('Date')
    return df

time_series_df = make_time_series(df)
print(time_series_df.head())
