import pandas as pd

data_path = './Data/[24-2 DS_Project2] Data.csv'
df = pd.read_csv(data_path)
cat_features = ['Gender', 'Card Brand', 'Card Type', 'Expires', 'Has Chip', 'Acct Open Date', 'Year PIN last Changed', 'Whether Security Chip is Used', 'Error Message', 'Month', 'Day']

# 각 범주형 변수의 고유값 개수 출력
print("\n=== 범주형 변수별 고유값 개수 ===")
for feature in cat_features:
    n_unique = df[feature].nunique()
    print(f"{feature}: {n_unique}개")
