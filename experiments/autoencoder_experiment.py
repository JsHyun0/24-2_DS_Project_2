import sys
import os
from tqdm import tqdm, trange

# 프로젝트 루트 디렉토리를 Python 경로에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

import neptune
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from Models.AutoEncoder import AutoEncoder, AE_validDataset, AE_trainDataset
from utils.utils import process_data
from sklearn.metrics import f1_score
import numpy as np

def objective(trial):
    # 1. 초기 설정
    run = neptune.init_run(
        project="jshyun/Datascience-Final",
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
    )
    
    # 실험 설정
    config = {
        "encoding_dim": trial.suggest_int("encoding_dim", 16, 32),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        "lr": trial.suggest_categorical("lr", [1e-3, 1e-4, 1e-5]),
        "epochs": 300,
        "threshold_percentile": trial.suggest_int("threshold_percentile", 90, 95),
        "l1_lambda": trial.suggest_float("l1_lambda", 1e-6, 1e-4, log=True),
    }
    
    # 데이터 준비
    cat_features = ['Gender', 'Card Brand', 'Card Type', 'Expires', 'Has Chip', 'Year PIN last Changed', 'Whether Security Chip is Used', 'Day', 'Error Message']

    num_features = ['Current Age', 'Retirement Age', 'Per Capita Income - Zipcode', 'Yearly Income', 'Total Debt', 'Credit Score', 'Credit Limit', 'Amount','Since Open Month']

    discarded = ['User', 'Birth Year', 'Birth Month', 'Card', 'Card Number', 'Zipcode', 'Merchandise Code', 'Acct Open Date', 'Year', 'Month']
    
    
    # 데이터 로드 및 전처리
    (train_cat_X, train_num_X, train_y), (valid_cat_X, valid_num_X, valid_y), label_encoders = process_data(
        os.path.join(PROJECT_ROOT, 'Data', '[24-2 DS_Project2] Data.csv'), 
        cat_features, 
        num_features,
        discarded
    )
    
    # 3. 모델 및 데이터로더 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터로더 초기화
    train_dataset = AE_trainDataset(train_cat_X, train_num_X, device)
    valid_dataset = AE_validDataset(valid_cat_X, valid_num_X, valid_y, device)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # 모델 초기화
    model = AutoEncoder(
        encoding_dim=config["encoding_dim"],
        cat_features=cat_features,
        num_features=num_features
    ).to(device)
    
    # Neptune 로깅 설정
    run["model/source_code"] = open(os.path.join(PROJECT_ROOT, "Models", "AutoEncoder.py"), "r").read()
    run["model/architecture"] = str(model)
    run["parameters"] = config
    
    # 학습 설정
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=20
    )
    criterion = nn.MSELoss()
    
    # 4. 학습 및 평가
    best_f1 = 0
    l1_lambda = config["l1_lambda"]
    model_filename = f"AE_dim{config['encoding_dim']}_batch{config['batch_size']}_lr{config['lr']:.6f}.pth"
    model_save_path = os.path.join(PROJECT_ROOT, "experiments", "AutoEncoder", model_filename)
    
    # 모델 저장 디렉토리 생성
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # 학습 루프 수정
    for epoch in trange(config["epochs"], desc="Training"):
        # 학습 단계
        model.train()
        train_loss = 0
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
        
        # 검증 단계 수정
        if epoch % 10 == 0:
            model.eval()
            valid_loss = 0
            reconstruction_errors = []
            all_labels = []
            
            with torch.no_grad():
                for cat_features, num_features, labels in tqdm(valid_loader, desc="Validation", leave=False):
                    y_hat, y = model(cat_features, num_features)
                    batch_loss = criterion(y_hat, y)
                    valid_loss += batch_loss.item()
                    
                    sample_errors = torch.mean((y_hat - y) ** 2, dim=1)
                    reconstruction_errors.extend(sample_errors.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                valid_loss /= len(valid_loader)
                
                # 성능 평가
                threshold = np.percentile(reconstruction_errors, config["threshold_percentile"])
                predictions = (np.array(reconstruction_errors) > threshold).astype(int)
                f1 = f1_score(all_labels, predictions)
                
                # scheduler 스텝 및 현재 학습률 로깅
                current_lr = optimizer.param_groups[0]['lr']
                scheduler.step(f1)
                new_lr = optimizer.param_groups[0]['lr']
                
                # 학습률이 변경되었을 때만 로그 출력
                if current_lr != new_lr:
                    print(f"Learning rate decreased from {current_lr} to {new_lr}")
                
                # 결과 로깅
                run["metrics/train_loss"].log(train_loss)
                run["metrics/valid_loss"].log(valid_loss)
                run["metrics/f1_score"].log(f1)
                run["metrics/learning_rate"].log(new_lr)
                print(f"Epoch {epoch}: Valid Loss = {valid_loss:.4f}, F1 Score = {f1:.4f}")
                
                # 최고 성능 모델 저장
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), model_save_path)
                    run["artifacts/best_model"].upload(model_save_path)
    
    run.stop()
    return best_f1

if __name__ == "__main__":
    # Optuna 학습 시작 (최소화 방향으로 설정)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    
    # 최적의 하이퍼파라미터 저장
    with open("experiments/best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4) 