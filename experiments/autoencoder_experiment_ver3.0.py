import sys
import os
from tqdm import tqdm, trange
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import neptune
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from Models.AutoEncoder import *
from utils.utils import *
from sklearn.metrics import f1_score
import numpy as np
from dotenv import load_dotenv
import time  # 파일 상단에 import 추가

# 프로젝트 루트 디렉토리를 Python 경로에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# .env 파일 로드
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

def objective(trial):
    # 1. 초기 설정
    run = neptune.init_run(
        project="jshyun/Datascience-Final",
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
    )
    
    # 실험 설정
    config = {
        "encoding_dim": trial.suggest_int("encoding_dim", 18, 32),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        "lr": 1e-4,
        "epochs": 500,
        "threshold_percentile": 95,
        "l1_lambda": trial.suggest_categorical("l1_lambda", [1e-4, 1e-5, 1e-6, ])
    }
    
    # 태그 추가 방식 수정
    run["sys/tags"].add("ver3.0")
    
    # 데이터 준비
    cat_features = [
        'Gender',
        'Zipcode',
        'Day',
        'Card Brand',
        'Card Type',
        'Has Chip',
        'Whether Security Chip is Used',
        'Error Message',
        'WeekDay',
        'Credit Signal',
        'PIN Change',
        'Security Level'
    ]
    num_features = [
        'Current Age',
        'Retirement Age',
        'Per Capita Income - Zipcode',
        'Yearly Income',
        'Total Debt',
        'Credit Score',
        'Valid Month',
        'Credit Limit',
        'Since Open Month',
        'Year PIN last Changed',
        'Amount',
        'Credit Util',
        'Years Changed PIN',
        'Security Score'
    ]
    discarded = [
        'User',
        'Birth Year',
        'Birth Month',
        'Year',
        'Month',
        'Merchandise Code',
        'Card',
        'Card Number',
        'Expires',
        'Acct Open Date',
    ]
    # 데이터 로드 및 전처리
    (train_cat_X, train_num_X, _), (valid_cat_X, valid_num_X, valid_y), _ , _= process_data(
        './Data/[24-2 DS_Project2] Data.csv', 
        cat_features, 
        num_features,
        discarded
    )
    
    # 3. 모델 및 데이터로더 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터로더 초기화
    train_dataset = AE_trainDataset(train_cat_X, train_num_X, device)
    valid_dataset = AE_trainDataset(valid_cat_X, valid_num_X, device)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # 모델 초기화
    model = AutoEncoder(
        encoding_dim=config["encoding_dim"],
        cat_features=cat_features,
        num_features=num_features
    ).to(device)
    
    # Kaiming Initialization 적용
    for name, param in model.named_parameters():
        if isinstance(param, nn.Linear):
            nn.init.kaiming_normal_(param.weight, mode='fan_in', nonlinearity='relu')
            if param.bias is not None:
                nn.init.zeros_(param.bias)
    
    # Neptune 로깅 설정
    run["model/source_code"] = open("Models/AutoEncoder.py", "r").read()
    run["model/architecture"] = str(model)
    run["parameters"] = config
    
    # 학습 설정
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.MSELoss()
    
    # 모델 저장 디렉토리 생성
    save_dir = "experiments/AutoEncoder3"
    os.makedirs(save_dir, exist_ok=True)
    
    # 4. 학습 및 평가
    best_loss = 0
    l1_lambda = config["l1_lambda"]
    model_filename = f"AE_dim{config['encoding_dim']}_batch{config['batch_size']}_lr{config['lr']:.6f}_l1{l1_lambda:.6f}.pth"
    
    # tqdm으로 에포크 진행률 표시
    for epoch in trange(config["epochs"], desc="Training"):
        # 학습 단계
        model.train()
        train_loss = 0
        # tqdm으로 배치 진행률 표시
        for cat_features, num_features in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            optimizer.zero_grad()
            y_hat, y = model(cat_features, num_features)
            
            # MSE 손실 계산
            mse_loss = criterion(y_hat, y)
            
            # L1 정규화 계산
            l1_reg = torch.tensor(0., requires_grad=True).to(device)
            for param in model.parameters():
                l1_reg = l1_reg + torch.norm(param, 1)
            
            # 총 손실 = MSE 손실 + L1 정규화
            loss = mse_loss + l1_lambda * l1_reg
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 평균 train_loss 계산
        train_loss /= len(train_loader)
        
        # 검증 단계 (10 에포크마다)
        if epoch % 10 == 0:
            run["metrics/train_loss"].log(train_loss)
            model.eval()
            valid_loss = 0
            reconstruction_errors = []
            all_labels = []
            
            with torch.no_grad():
                for cat_features, num_features in valid_loader:
                    y_hat, y = model(cat_features, num_features)
                    batch_loss = criterion(y_hat, y)
                    valid_loss += batch_loss.item()
                    
                    sample_errors = torch.mean((y_hat - y) ** 2, dim=1)
                    reconstruction_errors.extend(sample_errors.cpu().numpy())
                    #all_labels.extend(labels.cpu().numpy())
                
                # 성능 평가

                
                # 결과 로깅
                run["metrics/valid_loss"].log(valid_loss)

                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Valid Loss = {valid_loss:.4f}")
                
                # 최고 성능 모델 저장 및 혼동 행렬 생성
                if loss < best_loss:
                    best_loss = loss
                    model_path = os.path.join(save_dir, model_filename)
                    torch.save(model.state_dict(), model_path)
                    run["artifacts/best_model"].upload(model_path)


    
    run.stop()
    # Neptune 로그가 모두 전송될 때까지 대기
    time.sleep(3)
    return best_loss

if __name__ == "__main__":

    
    # Optuna 학습 시작 (최대화 방향으로 변경)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)
    
    # 최적의 하이퍼파라미터 저장
    with open("experiments/best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4) 