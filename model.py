import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """
    모델 구조 수정 금지.
    """
    def __init__(self, encoding_dim, cat_features, num_features, num_classes):
        super(BaseModel, self).__init__()
        self.cat_embeddings = nn.ModuleList([nn.Embedding(100, 5) for _ in cat_features])
        self.fc_cat = nn.Linear(len(cat_features) * 5 + len(num_features), 64)
        self.encoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.classifier = nn.Sequential(nn.Linear(32, 32),
                                        nn.ReLU(),
                                        nn.Linear(32, 16),
                                        nn.ReLU(),
                                        nn.Linear(16, num_classes))
    

    def forward(self, x_cat, x_num):
        embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        x = torch.cat(embeddings + [x_num], dim=1)
        x = self.fc_cat(x)
        encoded = self.encoder(x)
        out = self.classifier(encoded) # 이 레이어의 인풋을 활용할 것.
        return out

class myModel(BaseModel):
    def __init__(self, encoding_dim, cat_features, num_features, num_classes, dropout_rate=0.2):
        super(myModel, self).__init__(encoding_dim, cat_features, num_features, num_classes)
        #Decoder를 자유롭게 구현
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x_cat, x_num):
        embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        x = torch.cat(embeddings + [x_num], dim=1)
        x = self.fc_cat(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        out = self.classifier(decoded)
        return out

class GPUBaseModel(nn.Module):
    """
    모델 구조 수정 금지.
    """
    def __init__(self, encoding_dim, cat_features, num_features, num_classes, device):
        super(GPUBaseModel, self).__init__()
        self.cat_embeddings = nn.ModuleList([nn.Embedding(100, 5) for _ in cat_features])
        self.fc_cat = nn.Linear(len(cat_features) * 5 + len(num_features), 64)
        self.encoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.classifier = nn.Sequential(nn.Linear(32, 32),
                                        nn.ReLU(),
                                        nn.Linear(32, 16),
                                        nn.ReLU(),
                                        nn.Linear(16, num_classes))
        self.device = device

    def forward(self, x_cat, x_num):
        embeddings = [emb(x_cat[:, i].to(self.device)) for i, emb in enumerate(self.cat_embeddings)]
        x = torch.cat(embeddings + [x_num.to(self.device)], dim=1)
        x = self.fc_cat(x)
        encoded = self.encoder(x)
        out = self.classifier(encoded) # 이 레이어의 인풋을 활용할 것.
        return out