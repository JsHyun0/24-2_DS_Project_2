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


######## VER 4.0 ###################
### Dropout 추가하여 재구성 성능 향상
### 조건은 Fraud Label 다 제거하여 정상 데이터를 구축하는 성능을 향상하고자 함
### 또한 Embedding과 예측하도록 변경함
####################################
# .env 파일 로드
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

class AutoEncoder(BaseModel):
    def __init__(self, encoding_dim, cat_features, num_features, num_classes=1):
        super(AutoEncoder, self).__init__(encoding_dim, cat_features, num_features, num_classes)
        self.input_dim = len(cat_features)*5 + len(num_features)
        
        # Dropout 추가 및 더 깊은 네트워크 구성
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.input_dim)
        )

    def forward(self, x_cat, x_num):
        embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        x = torch.cat(embeddings + [x_num], dim=1)  # 임베딩된 데이터
        fc_x = self.fc_cat(x)
        encoded = self.encoder(fc_x)
        decoded = self.decoder(encoded)
        return decoded, x  # 임베딩된 데이터와 비교
    # 임베딩 추출, torch.eval() 모드에서 사용
    def get_embedding(self, x_cat, x_num):
        with torch.no_grad():
            embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
            original_x = torch.cat(embeddings + [x_num], dim=1)
            x = self.fc_cat(original_x)
            encoded = self.encoder(x)
        return encoded
 
def objective(trial):
    # 1. 초기 설정
    run = neptune.init_run(
        project="jshyun/Datascience-Final",
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
    )
    
    # 실험 설정 수정
    config = {
        "encoding_dim": trial.suggest_int("encoding_dim", 24, 36),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512]),
        "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),  # lr을 trial 파라미터로 변경
        "epochs": 300,
        "threshold_percentile": 95,
        "l1_lambda": trial.suggest_loguniform("l1_lambda", 1e-6, 1e-4)
    }
    
    # 태그 추가 방식 수정
    run["sys/tags"].add("ver4.0")
    
    # 데이터 준비
    ## Input Dimension : 12 + 14 = 26 
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
    valid_dataset = AE_validDataset(valid_cat_X, valid_num_X, valid_y, device)
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
    
    # scheduler 추가
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # 모델 저장 디렉토리 생성
    save_dir = "experiments/AutoEncoder4"
    os.makedirs(save_dir, exist_ok=True)
    
    # 4. 학습 및 평가
    best_loss = float('inf')
    l1_lambda = config["l1_lambda"]
    model_filename = f"AE4_dim{config['encoding_dim']}_batch{config['batch_size']}_lr{config['lr']:.6f}_l1{l1_lambda:.6f}.pth"
    
    # Early Stopping 추가
    early_stopping_patience = 10
    no_improve_count = 0
    
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
        
        # 검증 단계 수정 (매 에포크마다 검증 수행)
        model.eval()
        valid_loss = 0
        reconstruction_errors = []
        
        with torch.no_grad():
            for cat_features, num_features, labels in valid_loader:
                y_hat, y = model(cat_features, num_features)
                batch_loss = criterion(y_hat, y)
                valid_loss += batch_loss.item()
                
                # 재구성 오차 저장
                errors = torch.mean((y_hat - y) ** 2, dim=1)
                reconstruction_errors.extend(errors.cpu().numpy())
            
            valid_loss /= len(valid_loader)
            
            # Early Stopping 로직
            if valid_loss < best_loss:
                best_loss = valid_loss
                no_improve_count = 0
                # 모델 저장
                model_path = os.path.join(save_dir, model_filename)
                torch.save(model.state_dict(), model_path)
                run["artifacts/best_model"].upload(model_path)
            else:
                no_improve_count += 1
            
            if no_improve_count >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
            # Neptune 로깅 매 에포크마다 수행
            run["metrics/train_loss"].log(train_loss)
            run["metrics/valid_loss"].log(valid_loss)

            # scheduler step
            scheduler.step(valid_loss)
            
            # Neptune 로깅에 현재 learning rate 추가
            current_lr = optimizer.param_groups[0]['lr']
            run["metrics/learning_rate"].log(current_lr)

            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Valid Loss = {valid_loss:.4f}, LR = {current_lr:.6f}")
            

    
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

