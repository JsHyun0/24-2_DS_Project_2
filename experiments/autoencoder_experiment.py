import neptune
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
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
        "encoding_dim": trial.suggest_int("encoding_dim", 8, 32),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "lr": trial.suggest_loguniform("lr", 1e-4, 1e-2),
        "epochs": 100,
        "threshold": 0.2,  # 이상치 탐지를 위한 임계값
    }
    
    # 데이터 준비
    cat_features = ['Gender', 'Card Brand', 'Card Type', 'Expires', 'Has Chip', 
                   'Year PIN last Changed', 'Whether Security Chip is Used', 'Day']
    num_features = ['Current Age', 'Retirement Age', 'Per Capita Income - Zipcode', 
                   'Yearly Income', 'Total Debt', 'Credit Score', 'Credit Limit', 'Amount']
    
    # 데이터 로드 및 전처리
    (train_cat_X, train_num_X, _), (valid_cat_X, valid_num_X, valid_y) = process_data(
        './Data/[24-2 DS_Project2] Data.csv', 
        cat_features, 
        num_features
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
    run["model/source_code"] = open("Models/AutoEncoder.py", "r").read()
    run["model/architecture"] = str(model)
    run["parameters"] = config
    
    # 학습 설정
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.MSELoss()
    
    # 4. 학습 및 평가
    best_f1 = 0
    model_filename = f"AE_dim{config['encoding_dim']}_batch{config['batch_size']}_lr{config['lr']:.6f}.pth"
    
    for epoch in range(config["epochs"]):
        # 학습 단계
        model.train()
        for cat_features, num_features in train_loader:
            optimizer.zero_grad()
            y_hat, y = model(cat_features, num_features)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
        
        # 검증 단계 (10 에포크마다)
        if epoch % 10 == 0:
            run["metrics/train_loss"].log(loss.item())
            model.eval()
            reconstruction_errors = []
            all_labels = []
            
            with torch.no_grad():
                for cat_features, num_features, label in valid_loader:
                    y_hat, y = model(cat_features, num_features)
                    reconstruction_errors.extend(
                        torch.mean((y_hat - y) ** 2, dim=1).cpu().numpy()
                    )
                    all_labels.extend(label.cpu().numpy())
                
                # 성능 평가
                valid_loss = np.mean(reconstruction_errors)
                predictions = (np.array(reconstruction_errors) > config["threshold"]).astype(int)
                f1 = f1_score(all_labels, predictions)
                
                # 결과 로깅
                run["metrics/valid_loss"].log(valid_loss)
                run["metrics/f1_score"].log(f1)
                print(f"Epoch {epoch}: Valid Loss = {valid_loss:.4f}, F1 Score = {f1:.4f}")
                
                # 최고 성능 모델 저장
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), f"experiments/AutoEncoder/{model_filename}")
                    run["artifacts/best_model"].upload(f"experiments/AutoEncoder/{model_filename}")
    
    run.stop()
    return best_f1

if __name__ == "__main__":
    # Optuna 학습 시작 (최소화 방향으로 설정)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    
    # 최적의 하이퍼파라미터 저장
    with open("experiments/best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4) 