# 24-2_DS_Project_2
SSU 24-2 DataScience Final Project

## 프로젝트 설명
Fraud Detection 모델 학습 및 예측

## 프로젝트 환경
- Python 3.12.4
- PyTorch 2.1.1
- CUDA 12.1
> pip install -r pip_requirements.txt
> conda install --file conda_requirements.txt

## 프로젝트 구조
```
.
├── Data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── Models/
│   └── model.py
├── utils/
│   ├── util.py
│   └── args.py
├── experiments.py
├── main.py
├── model.py
```

## 실험 방법
```
python experiments.py
```
- 파라미터 범위 설정하기
```
study.suggest_loguniform('lr', 1e-4, 1e-2)
```

# utils.py Guide

## preprocess_data 호출
```
preprocess_data(
    data_path,
    cat_features,
    num_features,
    discarded,
)
```
return
(train_cat_x, train_num_x, train_y), 
(valid_cat_x, valid_num_x, valid_y), 
label_encoders

### 내부 동작
```
translation <- 필요한 컬럼 변환
in : df
out : df
```

```
iqr <- 이상치 제거
in : df, num_features
out : df
```

```
split <- train / valid 분리
in : df
out : train_df, valid_df
```

```
discard <- 불필요 컬럼 제거
in : df
out : df
```

```
scale <- num_features scaling
in: train_df, valid_df
out: train_df, valid_df
```

```
encode <- cat_features encode
in: train_df, valid_df
out: train_df, valid_df
```

```
target <- IsFraud 컬럼 반환
in : df
out : df['Is Fraud?']
```

```
unlabel <- train 데이터셋 Non Fraud 데이터만
in : df
out : df[df['Is Fraud?'] == 'No']
```
```
train cat/num 분리

valid cat/num 분리
```
### 사용 방법
```
train_cat_x, train_num_x, train_y, 
valid_cat_x, valid_num_x, valid_y, 
label_encoders = 
preprocess_data(data_path, cat_features, num_features, discarded)
```

