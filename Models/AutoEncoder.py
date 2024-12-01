import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Models.model import BaseModel
from utils.utils import process_data
from tqdm import tqdm
from sklearn.metrics import f1_score


class AutoEncoder(BaseModel):
    def __init__(self, encoding_dim, cat_features, num_features, num_classes=1):
        super(AutoEncoder, self).__init__(encoding_dim, cat_features, num_features, num_classes)
        self.input_dim = len(cat_features) + len(num_features)
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 48),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(),
            nn.Linear(48, self.input_dim)
        )

    def forward(self, x_cat, x_num):
        original_x = torch.cat([x_cat] + [x_num], dim=1)
        embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        x = torch.cat(embeddings + [x_num], dim=1)
        x = self.fc_cat(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, original_x
    # 임베딩 추출, torch.eval() 모드에서 사용
    def get_embedding(self, x_cat, x_num):
        with torch.no_grad():
            embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
            original_x = torch.cat(embeddings + [x_num], dim=1)
            x = self.fc_cat(original_x)
            encoded = self.encoder(x)
        return encoded
 

    
## Train Dataset
class AE_trainDataset(Dataset):
    def __init__(self, cat_features, num_features, device):
        self.cat_features = torch.tensor(cat_features.values, dtype=torch.long).to(device)
        self.num_features = torch.tensor(num_features.values, dtype=torch.float).to(device)
    def __len__(self):
        return len(self.cat_features)
    
    def __getitem__(self, idx):
        return self.cat_features[idx], self.num_features[idx]

## Valid Dataset
class AE_validDataset(Dataset):
    def __init__(self, cat_features, num_features, y, device):
        self.cat_features = torch.tensor(cat_features.values, dtype=torch.long).to(device)
        self.num_features = torch.tensor(num_features.values, dtype=torch.float).to(device)
        # y를 numpy array로 변환
        self.y = torch.tensor(y.values, dtype=torch.float).to(device)

    def __len__(self):
        return len(self.cat_features)
    
    def __getitem__(self, idx):
        return self.cat_features[idx], self.num_features[idx], self.y[idx]
    
class AE_Dataset(Dataset):
    def __init__(self, cat_features, num_features, y, device):
        self.cat_features = torch.tensor(cat_features.values, dtype=torch.long).to(device)
        self.num_features = torch.tensor(num_features.values, dtype=torch.float).to(device)
        self.y = torch.tensor(y.values, dtype=torch.float).to(device)

    def __len__(self):
        return len(self.cat_features)

    def __getitem__(self, idx):
        return self.cat_features[idx], self.num_features[idx], self.y[idx]
        

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    num_epochs = 100
    batch_size = 128
    lr = 1e-4
    # Feature Selection
    cat_features = ['Gender', 'Card Brand', 'Card Type', 'Expires', 'Has Chip', 'Year PIN last Changed', 'Whether Security Chip is Used', 'Day', 'Error Message']

    num_features = ['Current Age', 'Retirement Age', 'Per Capita Income - Zipcode', 'Yearly Income', 'Total Debt', 'Credit Score', 'Credit Limit', 'Amount','Since Open Month']

    discarded = ['User', 'Birth Year', 'Birth Month', 'Card', 'Card Number', 'Zipcode', 'Merchandise Code', 'Acct Open Date', 'Year', 'Month']
    # 프로젝트 루트 경로 설정
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, 'Data', '[24-2 DS_Project2] Data.csv')
    SAVE_PATH = os.path.join(PROJECT_ROOT, 'saved_models', 'AutoEncoder')

    # 데이터 로드 시
    (train_cat_X, train_num_X, train_y), (valid_cat_X, valid_num_X, valid_y), label_encoders = process_data(
        DATA_PATH,
        cat_features,
        num_features,
        discarded
    )
    train_dataset = AE_trainDataset(train_cat_X, train_num_X, device)
    valid_dataset = AE_validDataset(valid_cat_X, valid_num_X, valid_y, device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = AutoEncoder(encoding_dim=32, cat_features=cat_features, num_features=num_features).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)   
    criterion = nn.MSELoss()

    best_f1 = 0
    l1_lambda = 1e-5
    for epoch in tqdm(range(num_epochs), desc='학습 진행률', ncols=100, position=0, leave=True):
        # 학습 단계
        model.train()
        train_loss = 0
        for cat_features, num_features in train_loader:
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
    # ... existing code ...

        if epoch % 10 == 0:
            model.eval()
            valid_loss = 0
            reconstruction_errors = []
            all_labels = []
            
            with torch.no_grad():
                for cat_features, num_features, labels in valid_loader:
                    y_hat, y = model(cat_features, num_features)
                    batch_loss = criterion(y_hat, y)
                    valid_loss += batch_loss.item()
                    
                    sample_errors = torch.mean((y_hat - y) ** 2, dim=1)
                    reconstruction_errors.extend(sample_errors.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
                valid_loss /= len(valid_loader)
                
                threshold = np.percentile(reconstruction_errors, 95)
                predictions = (np.array(reconstruction_errors) > threshold).astype(int)
                f1 = f1_score(all_labels, predictions)
                
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_f1': best_f1,
                        'threshold': threshold
                    }, 'best_autoencoder.pth')
                    
                tqdm.write(f"에포크 {epoch}: Train Loss = {train_loss:.4f}, Valid Loss = {valid_loss:.4f}, F1 Score = {f1:.4f}")
        else:
            # 10 에포크마다가 아닐 때는 train_loss만 출력
            tqdm.write(f"에포크 {epoch}: Train Loss = {train_loss:.4f}")
