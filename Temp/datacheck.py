import pandas as pd
import torch

data_path = './Data/[24-2 DS_Project2] Data.csv'
df = pd.read_csv(data_path)

df = df.drop(['Year'])
