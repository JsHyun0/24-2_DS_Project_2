import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader

sys.path.append('../')
from model import BaseModel
from util import process_data

class AutoEncoder(BaseModel):
    def __init__(self, encoding_dim, cat_features, num_features, num_classes):
        super(AutoEncoder, self).__init__(encoding_dim, cat_features, num_features, num_classes)
        self.input_dim = len(cat_features) * 5 + len(num_features)
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.input_dim),
        )
        self.reconstruction_layer = nn.Sequential(
            nn.Linear(64, self.input_dim),
            nn.ReLU()
        )
    def forward(self, x_cat, x_num):
        embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        original_x = torch.cat(embeddings + [x_num], dim=1)
        x = self.fc_cat(original_x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        reconstructed = self.reconstruction_layer(decoded)
        return reconstructed, original_x
    
    # 임베딩 추출, torch.eval() 모드에서 사용
    def get_embedding(self, x_cat, x_num):
        embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        original_x = torch.cat(embeddings + [x_num], dim=1)
        x = self.fc_cat(original_x)
        encoded = self.encoder(x)
        return encoded

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cat_features = ['Gender', 'Card Brand', 'Card Type', 'Expires', 'Has Chip', 'Year PIN last Changed', 'Whether Security Chip is Used', 'Error Message', 'Month', 'Day']
    num_features = ['Current Age', 'Retirement Age', 'Per Capita Income - Zipcode', 'Yearly Income', 'Total Debt', 'Credit Score', 'Credit Limit', 'Amount']
    data = pd.read_csv('../Data/[24-2 DS_Project2] Data.csv')
    (train_cat_X, train_num_X, _), (valid_cat_X, valid_num_X, _) = process_data(data, cat_features, num_features)

    # Custom Dataset 클래스 정의
    class AutoEncoderDataset(Dataset):
        def __init__(self, cat_features, num_features):
            self.cat_features = torch.tensor(cat_features.values, dtype=torch.long).to(device)
            self.num_features = torch.tensor(num_features.values, dtype=torch.float).to(device)
            
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return self.cat_features[idx], self.num_features[idx]

    # 학습용, 검증용 데이터셋 생성
    train_dataset = AutoEncoderDataset(train_cat_X, train_num_X)
    valid_dataset = AutoEncoderDataset(valid_cat_X, valid_num_X)


    # 데이터 로더 (배치 생성)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

    model = AutoEncoder(encoding_dim=64, cat_features=cat_features, num_features=num_features, num_classes=1)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(300):
        model.train()
        train_loss = 0
        for batch_idx, (cat_features, num_features) in enumerate(train_loader):
            optimizer.zero_grad()
            output, original_x = model(cat_features, num_features)
            loss = criterion(output, original_x)
            loss.backward()
            optimizer.step()

