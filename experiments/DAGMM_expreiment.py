import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


from tqdm import tqdm, trange
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
import neptune
import optuna
import torch
import torch.nn as nn   
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from Models.DAGMM import *
from Models.AutoEncoder import *
from utils.utils import *
import numpy as np
from dotenv import load_dotenv
import time  # 파일 상단에 import 추가

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
        "batch_size": 1024,
        "lr": 1e-4,
        "epochs": 300,
        "threshold_percentile": 95,
        "n_gmm": trial.suggest_categorical("n_gmm", [4, 8, 10]),
        "lambda_energy": 0.1,
        "lambda_cov_diag" : 0.005,
    }
    
    # 태그 추가
    run["sys/tags"].add("DAGMM_1.0")
    
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
    (train_cat_X, train_num_X, train_y), (valid_cat_X, valid_num_X, valid_y), _ , _= process_data(
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
    model = DAGMM(
        encoding_dim=config["encoding_dim"],
        n_gmm=config["n_gmm"],
        cat_features=cat_features,
        num_features=num_features,
    ).to(device)
    
    # Kaiming Initialization 적용
    # for name, param in model.named_parameters():
    #     if isinstance(param, nn.Linear):
    #         nn.init.kaiming_normal_(param.weight, mode='fan_in', nonlinearity='relu')
    #         if param.bias is not None:
    #             nn.init.zeros_(param.bias)
    
    # Neptune 로깅 설정
    run["model/architecture"] = str(model)
    run["parameters"] = config
    
    # 학습 설정
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',           # 검증 손실 감소 방향으로 최적화
        factor=0.2,           # 학습률 50% 감소
        patience=5,          # 5 에포크 동안 개선이 없으면 학습률 조정
        verbose=True,         # 학습률 변경 시 메시지 출력
        min_lr=1e-6          # 최소 학습률
    )
    
    # 모델 저장 디렉토리 생성
    save_dir = "experiments/DAGMM"
    os.makedirs(save_dir, exist_ok=True)
    
    # 4. 학습 및 평가
    model_filename = f"DAGMM_dim{config['encoding_dim']}_batch{config['batch_size']}_lr{config['lr']:.6f}_n-gmm{config['n_gmm']}.pth"

    best_f1 = 0

    # tqdm으로 에포크 진행률 표시
    for epoch in trange(config["epochs"], desc="Training", position=0, leave=True):
        # 학습 단계
        model.train()
        train_loss = 0
        # tqdm으로 배치 진행률 표시
        for cat_features, num_features in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, position=1):
            
            x, enc, dec, z, gamma = model(cat_features, num_features)
            # MSE 손실 계산
            loss, recon_loss, energy, cov_diag = model.loss(x, dec, z, gamma, lambda_energy=config["lambda_energy"], lambda_cov_diag=config["lambda_cov_diag"] )
            
            model.zero_grad()
            loss = loss.mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        
        # 검증 단계 (5 에포크마다)
        if epoch % 5 == 0:
            run["metrics/train_loss"].log(train_loss)
            model.eval()
            valid_loss = 0
            reconstruction_errors = []
            all_labels = []
            all_z = []  # 잠재 벡터 저장
            
            with torch.no_grad():
                # 1. 학습 데이터로 GMM 파라미터 한 번만 계산
                N = 0
                mu_sum = 0
                cov_sum = 0
                gamma_sum = 0
                
                for cat_features, num_features in train_loader:
                    x, enc, dec, z, gamma = model(cat_features, num_features)
                    # GMM 파라미터 계산
                    phi, mu, cov = model.compute_gmm_params(z, gamma)
                    
                    batch_gamma_sum = torch.sum(gamma, dim=0)
                    gamma_sum += batch_gamma_sum
                    mu_sum += mu * batch_gamma_sum.unsqueeze(-1)
                    cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)
                    N += x.size(0)
                
                # GMM 파라미터 계산
                train_phi = gamma_sum / N
                train_mu = mu_sum / gamma_sum.unsqueeze(-1)
                train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)
                
                # 2. 검증 데이터에 대한 평가
                valid_loss = 0
                reconstruction_errors = []
                all_labels = []
                all_z = []
                
                for cat_features, num_features, labels in valid_loader:
                    x, enc, dec, z, gamma = model(cat_features, num_features)
                    
                    # 손실 계산 추가
                    loss, recon_loss, energy, cov_diag = model.loss(
                        x, dec, z, gamma, 
                        lambda_energy=config["lambda_energy"], 
                        lambda_cov_diag=config["lambda_cov_diag"]
                    )
                    valid_loss += loss.mean().item()
                    
                    # 저장된 GMM 파라미터를 사용하여 에너지 계산
                    sample_energy = model.compute_energy(
                        z, 
                        phi=train_phi,
                        mu=train_mu, 
                        cov=train_cov
                    )
                    
                    reconstruction_errors.extend(sample_energy.cpu().numpy())
                    all_z.extend(z.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                # 배치 평균 valid_loss 계산
                valid_loss /= len(valid_loader)

                # 3. 이상치 탐지
                reconstruction_errors = np.array(reconstruction_errors)
                all_labels = np.array(all_labels)
                
                # 상위 10%를 이상치로 판단
                threshold = np.percentile(reconstruction_errors, 90)
                predictions = (reconstruction_errors > threshold).astype(int)
                
                # 성능 평가
                f1 = f1_score(all_labels, predictions)
                recall = recall_score(all_labels, predictions)
                precision = precision_score(all_labels, predictions, zero_division=1)
                accuracy = accuracy_score(all_labels, predictions)
                
                # 결과 로깅
                run["metrics/valid_loss"].log(valid_loss)
                run["metrics/accuracy"].log(accuracy)
                run["metrics/recall"].log(recall)
                run["metrics/precision"].log(precision)
                run["metrics/f1_score"].log(f1)
                run["metrics/threshold"].log(threshold)

                # 4. 스케줄러 업데이트 및 현재 학습률 로깅
                scheduler.step(f1)
                current_lr = optimizer.param_groups[0]['lr']
                run["metrics/learning_rate"].log(current_lr)
                
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                      f"Valid Loss = {valid_loss:.4f}, "
                      f"F1 Score = {f1:.4f}, "
                      f"LR = {current_lr:.6f}")
                
                # 최고 성능 모델 저장 및 혼동 행렬 생성
                if f1 > best_f1:
                    best_f1 = f1
                    model_path = os.path.join(save_dir, model_filename)
                    torch.save(model.state_dict(), model_path)
                    run["artifacts/best_model"].upload(model_path)
                    print(f"새로운 최고 성능 모델 저장됨 (F1 Score: {f1:.4f})")


    
    run.stop()
    # Neptune 로그가 모두 전송될 때까지 대기
    time.sleep(3)
    return best_f1

if __name__ == "__main__":

    
    # Optuna 학습 시작 (최대화 방향으로 변경)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    
    # 최적의 하이퍼파라미터 저장
    with open("experiments/best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4) 