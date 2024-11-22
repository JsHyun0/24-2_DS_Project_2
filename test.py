import model
import torch
import util
import args
import torch.nn as nn
from torch.utils.data import DataLoader
## 여기서 본인이 만든 모델 및 데이터 전처리를 메챠쿠차 테스트하시길 바랍니다.

# 데이터 전처리
data_path = './Data/[24-2 DS_Project2] Data.csv'
train_data, valid_data, cat_features, num_features = util.process_data(data_path)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)

# 학습 세팅
basemodel = model.BaseModel(32, cat_features, num_features, 2)
optimizer = torch.optim.Adam(basemodel.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# 학습
for epoch in range(100):
    for batch in train_loader:
        x_cat, x_num, y = batch
        y_pred = basemodel(x_cat, x_num)
        loss = criterion(y_pred[0], y) + criterion(y_pred[1], y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        # validation loss 계산 - 10 에폭마다 실행
        basemodel.eval()
        valid_loss = 0
        with torch.no_grad():
            for valid_batch in valid_loader:
                x_cat_valid, x_num_valid, y_valid = valid_batch
                y_pred_valid = basemodel(x_cat_valid, x_num_valid)
                valid_loss += criterion(y_pred_valid[0], y_valid) + criterion(y_pred_valid[1], y_valid)
        valid_loss /= len(valid_loader)
        print(f'Epoch {epoch}, Train loss: {loss.item()}, Validation Loss: {valid_loss}')
        basemodel.train()
            

# 검증

    

