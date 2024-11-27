import model
import torch
import util
import args
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
## 여기서 본인이 만든 모델 및 데이터 전처리를 메챠쿠차 테스트하시길 바랍니다.

# Feature Selection
cat_features = ['Gender', 'Card Brand', 'Card Type', 'Expires', 'Has Chip', 'Year PIN last Changed', 'Whether Security Chip is Used', 'Error Message', 'Month', 'Day']
num_features = ['Current Age', 'Retirement Age', 'Per Capita Income - Zipcode', 'Yearly Income', 'Total Debt', 'Credit Score', 'Credit Limit', 'Amount']

# 데이터 전처리
data_path = './Data/[24-2 DS_Project2] Data.csv'
(train_cat_X, train_num_X, train_y), (valid_cat_X, valid_num_X, valid_y) = util.process_data(data_path, cat_features, num_features)

# device 설정을 클래스 정의 전에 이동
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용 중인 디바이스: {device}')

# Custom Dataset 클래스 정의
class FraudDataset(torch.utils.data.Dataset):
    def __init__(self, cat_features, num_features, labels):
        self.cat_features = torch.tensor(cat_features.values, dtype=torch.long).to(device)
        self.num_features = torch.tensor(num_features.values, dtype=torch.float).to(device)
        self.labels = torch.tensor(labels.values, dtype=torch.float).to(device)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.cat_features[idx], self.num_features[idx], self.labels[idx]

# 학습용, 검증용 데이터셋 생성
train_dataset = FraudDataset(train_cat_X, train_num_X, train_y)
valid_dataset = FraudDataset(valid_cat_X, valid_num_X, valid_y)


# 데이터 로더 (배치 생성)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)  # 검증은 shuffle 불필요

# 학습 세팅
basemodel = model.GPUBaseModel(32, cat_features, num_features, 1, device)
basemodel = basemodel.to(device)
optimizer = torch.optim.Adam(basemodel.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# 학습
best_valid_loss = float('inf')
train_losses = []
valid_losses = []

for epoch in range(100):
    # 학습
    basemodel.train()
    train_loss = 0
    for batch in tqdm(train_loader):
        x_cat, x_num, y = [b.to(device) for b in batch]
        y = y.float().view(-1, 1)
        y_pred = basemodel(x_cat, x_num)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # 검증
    if epoch % 10 == 0:
        basemodel.eval()
        valid_loss = 0
        with torch.no_grad():
            for valid_batch in tqdm(valid_loader):
                x_cat_valid, x_num_valid, y_valid = [b.to(device) for b in valid_batch]
                y_valid = y_valid.float().view(-1, 1)
                y_pred_valid = basemodel(x_cat_valid, x_num_valid)
                valid_loss += criterion(y_pred_valid, y_valid)
        
        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)
        
        print(f'Epoch {epoch}, Train loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')
        
        # 모델 저장
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(basemodel.state_dict(), 'best_model.pth')

        basemodel.train()

# 검증

    

