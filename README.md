# 24-2_DS_Project_2
SSU 24-2 DataScience Final Project

## 프로젝트 설명
Fraud Detection 모델 학습 및 예측

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