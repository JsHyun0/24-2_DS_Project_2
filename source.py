import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from typing import Tuple, List
from joblib import dump

## ----------------------------------PREPROCESS
# Constant Definition
TRAIN_MONTH_START = 1
TRAIN_MONTH_END = 9
IQR_MULTIPLIER = 1.5
NONE_LABEL = 'None'

def SelectFeature():

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
    return cat_features, num_features, discarded
# need to separate by models
def evaluate_validation(model, data):
    model.eval()
    with torch.no_grad():
        y_pred = model(data)
        return y_pred

def evaluate_test(model, data):
    model.eval()
    with torch.no_grad():
        y_pred = model(data)
        return y_pred


# Data Preprocess
def zipcode(df) :
    trans = df.copy()
    trans['Zipcode'] = (trans['Zipcode'] // 10000).astype(str)
    return trans

def convert_date_to_weekday(df, col_Year, col_Month, col_Day):

    # 날짜 컬럼 생성
    df['WeekDay'] = pd.to_datetime(
        df[col_Year].astype(str) + '-' +
        df[col_Month].astype(str).str.zfill(2) + '-' +
        df[col_Day].astype(str).str.zfill(2)
    )

    # 요일 추출 (0=월요일, 6=일요일)
    df['WeekDay'] = df['WeekDay'].dt.dayofweek

    # 요일 이름으로 변환
    weekday_map = {
        0: 'MON',
        1: 'TUE',
        2: 'WED',
        3: 'THU',
        4: 'FRI',
        5: 'SAT',
        6: 'SUN'
    }

    df['WeekDay'] = df['WeekDay'].map(weekday_map)

    return df

def security_level(df: pd.DataFrame) -> pd.DataFrame:
    df['Years Changed PIN'] = df['Year'] - df['Year PIN last Changed']

    # PIN 변경 기간에 따른 점수 조건
    conditions = [
        (df['Years Changed PIN'] >= 5),
        (df['Years Changed PIN'] < 5),
        (df['Years Changed PIN'] < 3),
        (df['Years Changed PIN'] < 1),
        (df['Years Changed PIN'] < 0)
    ]

    choices = [0, 0.25, 0.5, 0.75, 1]

    level = ['BAD', "LOW", "NORMAL", "HIGH", "SECURE"]

    # PIN 점수 계산
    df['PIN Change'] = np.select(conditions, choices, default=0)

    # 최종 보안 레벨 계산
    df['Security Score'] = (df['Has Chip'].astype(int) * 0.3 + df['PIN Change'] * 0.7)

    # 조건을 내림차순으로 변경
    levels = [
        (df['Security Score'] >= 0.8),
        (df['Security Score'] >= 0.6),
        (df['Security Score'] >= 0.4),
        (df['Security Score'] >= 0.2),
        (df['Security Score'] >= 0)
    ]

    df['Security Level'] = np.select(levels, level[::-1], default="BAD")  # level 리스트도 역순으로 적용

    return df

def credit_util(df: pd.DataFrame) -> pd.DataFrame:
    df['Credit Util'] = df['Amount'] / df['Credit Limit']
    df['Credit Util'] = np.where(np.isinf(df['Credit Util']), 1, df['Credit Util'])
    df['Credit Util'] = np.where(df['Credit Limit'] == 0, 1, df['Credit Util'])
    df['Credit Util'] = np.where(df['Amount'] == 0, 1, df['Credit Util'])
    conditions = [
        (df['Credit Util'] <= 0.2),
        (df['Credit Util'] <= 0.5),
        (df['Credit Util'] <= 0.8),
        (df['Credit Util'] > 0.8)
    ]

    choices = ['NORMAL', 'CAUTION', 'WARNING', 'DANGER']

    df['Credit Signal'] = np.select(conditions, choices, default='DANGER')

    return df

## feature modify
def translation(df: pd.DataFrame) -> pd.DataFrame :
    if 'Acct Open Date' not in df.columns :
        raise ValueError("TRANSITION ValueError : 'Acct Open Date' COLUMN DOES NOT EXIST")

    if 'Error Message' not in df.columns :
        raise ValueError("TRANSITION ValueError : 'Error Message' COLUMN DOES NOT EXIST")

    if 'Year' not in df.columns :
        raise ValueError("Year ValueError : 'Year' COLUMN DOES NOT EXIST")

    if 'Month' not in df.columns :
        raise ValueError("Month ValueError : 'Month' COLUMN DOES NOT EXIST")

    if 'Day' not in df.columns :
        raise ValueError("Day ValueError : 'Day' COLUMN DOES NOT EXIST")

    trans = df.copy()

    # Error Message -> bool
    trans['Error Message'] = trans['Error Message'].astype(bool)
    # zipcode mapping by Administrative district
    trans = zipcode(trans)
    # Add Transaction Week
    trans = convert_date_to_weekday(trans, 'Year', 'Month', 'Day')
    # Add Months from Account Open Date
    trans['Since Open Month'] = (trans['Year'] - trans['Acct Open Date'].str[-4:].astype(int)) * 12 + (trans['Month'] - trans['Acct Open Date'].str[:2].astype(int)).astype(int)
    # Add Valid Months
    trans['Valid Month'] = (trans['Expires'].str[-4:].astype(int) - trans['Year']) * 12 + (trans['Expires'].str[:2].astype(int)- trans['Month'].astype(int))
    # Add Credit Utilization
    trans = credit_util(trans)
    # Add Security Level
    trans = security_level(trans)

    # whole dataset
    return trans

# MUST BE EXECUTED AFTER TRANSITION
def discard(df: pd.DataFrame, discarded : List[str]) -> pd.DataFrame :
    if not all(feature in df.columns for feature in discarded) :
        raise ValueError("DISCARD ValueError : NO COLUMN IN GIVEN DATASET")

    # whole dataset
    return df.drop(columns=discarded)

## Train-Valid Set split
def split_by_date(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame] :
    # Value Error : There is no any Month in given DATASET
    if 'Month' not in df.columns :
        raise ValueError("SPLIT ValueError : 'Month' COLUMN DOES NOT EXIST.")

    mask = df['Month'].between(TRAIN_MONTH_START, TRAIN_MONTH_END)
    train_df = df[mask]
    val_df = df[~mask]


    return train_df, val_df

## num_features
def iqr(df: pd.DataFrame, num_features: List[str]) -> pd.DataFrame :
    # ValueError - NO COLUMN
    if not all(feature in df.columns for feature in num_features) :
        raise ValueError("IQR ValueError : NO COLUMN IN GIVEN DATASET")

    num_df = df.copy()
    mask = pd.Series(True, index=df.index)

    for col in num_features :
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        l_bound = Q1 - IQR_MULTIPLIER * IQR
        u_bound = Q3 + IQR_MULTIPLIER * IQR
        num_df = num_df[(num_df[col] >= l_bound) & (num_df[col] <= u_bound)]

    return num_df

def scale(train_df: pd.DataFrame, valid_df: pd.DataFrame, num_features: List[str]) -> Tuple[pd.DataFrame , pd.DataFrame, StandardScaler] :
    # ValueError - NO COLUMN
    if not all(feature in train_df.columns for feature in num_features) :
        raise ValueError("SCALE ValueError : NO COLUMN IN GIVEN DATASET")
    if not all(feature in valid_df.columns for feature in num_features) :
        raise ValueError("SCALE ValueError : NO COLUMN IN GIVEN DATASET")

    scaler = StandardScaler()
    scaler.fit(train_df[num_features])

    train_df[num_features] = pd.DataFrame(
        scaler.transform(train_df[num_features]),
        columns = num_features,
        index = train_df.index,
    )
    valid_df[num_features] = pd.DataFrame(
        scaler.transform(valid_df[num_features]),
        columns = num_features,
        index = valid_df.index,
    )

    return train_df, valid_df, scaler

## cat_features
def encode(train_df: pd.DataFrame, valid_df: pd.DataFrame, cat_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, any] :
    # ValueError - NO COLUMN
    if not all(feature in train_df.columns for feature in cat_features) :
        raise ValueError("ENCODE ValueError : NO COLUMN IN GIVEN DATASET")

    label_encoders = {}

    for col in cat_features :
        le = LabelEncoder()
        le.fit(train_df[col].astype(str))
        train_df[col] = le.transform(train_df[col].astype(str))
        valid_df[col] = le.transform(valid_df[col].astype(str))
        label_encoders[col] = le

    return train_df, valid_df, label_encoders

def get_target(df: pd.DataFrame) -> pd.DataFrame :
    if 'Is Fraud?' not in df.columns:
        raise ValueError("GET_TARGET ValueError : NO TARGET IN GIVEN DATASET")

    target = pd.DataFrame(df['Is Fraud?'] == 'Yes').astype(float)

    return target

def discard_label(df: pd.DataFrame) -> pd.DataFrame :
    if 'Is Fraud?' not in df.columns:
        raise ValueError("DISCARD_TARGET ValueError : NO TARGET IN GIVEN DATASET")

    unlabeled = df[df['Is Fraud?'] == "No"]

    return unlabeled

def process_data(data_path: str, cat_features: List[str], num_features: List[str], discarded: List[str]) -> Tuple:
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CAN NOT FIND DATA AT : {data_path}")
    except Exception as e:
        raise Exception(f"ERROR RAISED DURING LOADING DATA: {str(e)}")

    if df.empty:
        raise ValueError("ValueError : EMPTY DATASET")

    if 'Is Fraud?' not in df.columns:
        raise ValueError("ValueError : NO TARGET IN GIVEN DATASET")

    label_encoders = {}

    print("TRANSITION")
    df = translation(df)
    trans = df.copy()

    print("IQR")
    df = iqr(df, num_features)

    print("SPLIT")
    train_df, valid_df = split_by_date(df)

    print("DISCARD")
    train_df = discard(train_df, discarded)
    valid_df = discard(valid_df, discarded)

    print("SCALE")
    train_df, valid_df, scaler = scale(train_df, valid_df, num_features)

    print("ENCODE")
    train_df, valid_df, label_encoders = encode(train_df, valid_df, cat_features)

    print("UNLABEL")
    train_df = discard_label(train_df)

    print("TARGET")
    train_y = get_target(train_df)
    valid_y = get_target(valid_df)

    print("TRAIN CAT/NUM")
    train_cat_x = train_df[cat_features]
    train_num_x = train_df[num_features]

    print("VALID CAT/NUM")
    valid_num_x = valid_df[num_features]
    valid_cat_x = valid_df[cat_features]

    print("RETURN")
    return ((train_cat_x, train_num_x, train_y),
            (valid_cat_x, valid_num_x, valid_y),
            label_encoders, trans, scaler)

def dt_process_data(data_path: str, cat_features: List[str], num_features: List[str], discarded: List[str]) -> Tuple:
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CAN NOT FIND DATA AT : {data_path}")
    except Exception as e:
        raise Exception(f"ERROR RAISED DURING LOADING DATA: {str(e)}")

    if df.empty:
        raise ValueError("ValueError : EMPTY DATASET")

    if 'Is Fraud?' not in df.columns:
        raise ValueError("ValueError : NO TARGET IN GIVEN DATASET")

    label_encoders = {}

    print("TRANSITION")
    df = translation(df)
    trans = df.copy()

    print("IQR")
    df = iqr(df, num_features)

    print("SPLIT")
    train_df, valid_df = split_by_date(df)

    print("DISCARD")
    train_df = discard(train_df, discarded)
    valid_df = discard(valid_df, discarded)

    print("SCALE")
    train_df, valid_df, scaler = scale(train_df, valid_df, num_features)

    print("ENCODE")
    train_df, valid_df, label_encoders = encode(train_df, valid_df, cat_features)

    print("TARGET")
    train_y = get_target(train_df)
    valid_y = get_target(valid_df)

    print("TRAIN CAT/NUM")
    train_cat_x = train_df[cat_features]
    train_num_x = train_df[num_features]

    print("VALID CAT/NUM")
    valid_num_x = valid_df[num_features]
    valid_cat_x = valid_df[cat_features]

    print("RETURN")
    return ((train_cat_x, train_num_x, train_y),
            (valid_cat_x, valid_num_x, valid_y),
            label_encoders, trans, scaler)



## ----------------------------------MODELS
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

class AutoEncoder(BaseModel):
    def __init__(self, encoding_dim, cat_features, num_features, num_classes=1):
        super(AutoEncoder, self).__init__(encoding_dim, cat_features, num_features, num_classes)
        self.input_dim = len(cat_features)*5 + len(num_features)

        # Dropout 추가 및 더 깊은 네트워크 구성
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
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

class DAGMM(BaseModel):
    def __init__(self, encoding_dim, n_gmm, cat_features, num_features, num_classes=1):
        super(DAGMM, self).__init__(encoding_dim, cat_features, num_features, num_classes)
        self.input_dim = len(cat_features) + len(num_features)
        self.AEDecoder = nn.Sequential(
            nn.Linear(encoding_dim, 48),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(),
            nn.Linear(48, self.input_dim)
        )

        self.estimation = nn.Sequential(
            nn.Linear(encoding_dim + 1, 10),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(10, n_gmm),
            nn.Softmax(dim=1)
        )

        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm, encoding_dim + 1))
        self.register_buffer("cov", torch.zeros(n_gmm, encoding_dim + 1, encoding_dim + 1))

    def euclidian_distance(self, x, y):
        return (x-y).norm(2, dim=1) / x.norm(2, dim=1)

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)  # 샘플 개수
        sum_gamma = torch.sum(gamma, dim=0)

        # GMM 매개변수 업데이트
        phi = sum_gamma / N
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)

        self.phi = phi.data
        self.mu = mu.data
        self.cov = cov.data

        return phi, mu, cov

    def compute_energy(self, z, phi, mu, cov):
        eps = 1e-12
        k, D, _ = cov.size()
        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)

        cov_inv = []
        det_cov = []
        cov_diag = 0

        for i in range(k):
            cov_k = cov[i] + torch.eye(D).to(cov.device) * eps
            cov_inv.append(torch.inverse(cov_k).unsqueeze(0))

            # Determinant of covariance
            det_cov.append(torch.det(cov_k).unsqueeze(0))

            # Diagonal of covariance
            cov_diag += torch.sum(1 / cov_k.diagonal())

        cov_inv = torch.cat(cov_inv, dim=0)
        det_cov = torch.cat(det_cov).cuda()



        exp_term_tmp = -0.5 * torch.sum(
            torch.sum(z_mu.unsqueeze(-1) * cov_inv.unsqueeze(0), dim=-2) * z_mu, dim=-1
        )
        # Stabilize with max_val
        max_val = torch.max(exp_term_tmp, dim=1, keepdim=True)[0]
        exp_term = torch.exp(exp_term_tmp - max_val)  # Shape: (N, K)

        # Compute log probabilities
        log_term = torch.log(phi + eps) - 0.5 * torch.log(det_cov + eps) - 0.5 * D * torch.log(torch.tensor(2 * np.pi).to(cov.device))
        log_prob = exp_term + log_term.unsqueeze(0)  # Shape: (N, K)

        # Energy
        energy = -torch.logsumexp(log_prob, dim=1)  # Shape: (N,)
        return energy

    # 임베딩 추출, torch.eval() 모드에서 사용
    def get_embedding(self, x_cat, x_num):
        with torch.no_grad():
            embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
            original_x = torch.cat(embeddings + [x_num], dim=1)
            x = self.fc_cat(original_x)
            encoded = self.encoder(x)
        return encoded
    def get_embedding_cat(self, x_cat):
        with torch.no_grad():
            embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        return embeddings
    def mse_reconstruction_error(self, x, x_hat):
        return torch.mean((x - x_hat) ** 2, dim=1)  # 샘플별 MSE
    # 학습과정
    def forward(self, x_cat, x_num):
        original_x = torch.cat([x_cat]+ [x_num], dim=1)
        embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        x = torch.cat(embeddings + [x_num], dim=1)

        enc = self.encoder(self.fc_cat(x))

        dec = self.AEDecoder(enc)


        # reconsturction error 구하기
        rec_mse = self.mse_reconstruction_error(original_x, dec)
        #rec_cosine = F.cosine_similarity(x, dec, dim=1)
        #rec_euclidian = self.euclidian_distance(x, dec)

        z = torch.cat([enc, rec_mse.unsqueeze(-1)], dim=1)

        gamma = self.estimation(z)

        return original_x, enc, dec, z, gamma

    def loss(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):
        # 복원 손실 계산
        recon_loss = self.mse_reconstruction_error(x, x_hat)
        l1_reg = sum(torch.norm(p, 1) for p in self.parameters())
        # GMM 매개변수 계산
        phi, mu, cov = self.compute_gmm_params(z, gamma)

        # 에너지 계산
        energy = self.compute_energy(z, phi, mu, cov)

        # 공분산 정규화
        cov_diag = torch.mean(torch.diagonal(cov, dim1=-2, dim2=-1))

        # 총 손실
        total_loss = recon_loss + lambda_energy * energy + lambda_cov_diag * cov_diag + + 1e-5 * l1_reg
        return total_loss, recon_loss, energy, cov_diag



## ----------------------------------DATASETS
class AE_trainDataset(Dataset):
    def __init__(self, cat_features, num_features, device):
        self.cat_features = torch.tensor(cat_features.values, dtype=torch.long).to(device)
        self.num_features = torch.tensor(num_features.values, dtype=torch.float).to(device)
    def __len__(self):
        return len(self.cat_features)

    def __getitem__(self, idx):
        return self.cat_features[idx], self.num_features[idx]
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


def TrainAE(model, config, train_data):
    ae_filename = "autoencoder.pth"
    cat_features, num_features, discarded = SelectFeature()
    (train_cat_X, train_num_X, train_y), (valid_cat_X, valid_num_X, valid_y), _ , _, _= process_data(
        train_data,
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

    # Kaiming Initialization 적용
    for name, param in model.named_parameters():
        if isinstance(param, nn.Linear):
            nn.init.kaiming_normal_(param.weight, mode='fan_in', nonlinearity='relu')
            if param.bias is not None:
                nn.init.zeros_(param.bias)
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

    # 4. 학습 및 평가
    best_loss = float('inf')
    l1_lambda = config["l1_lambda"]

    # Early Stopping 추가
    early_stopping_patience = 18
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
                torch.save(model.state_dict(), ae_filename)
            else:
                no_improve_count += 1

            if no_improve_count >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

            # scheduler step
            scheduler.step(valid_loss)

            # Neptune 로깅에 현재 learning rate 추가
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Valid Loss = {valid_loss:.4f}, LR = {current_lr:.6f}")

def TrainDT(model, config, train_data):
    dt_filename = "encodedtree.joblib"
    cat_features, num_features, discarded = SelectFeature()
    (train_cat_X, train_num_X, train_y), (valid_cat_X, valid_num_X, valid_y), le , _, scaler= dt_process_data(
        train_data,
        cat_features,
        num_features,
        discarded
    )
    (_, _, _), (_, _, _), le_deep , _, scaler_deep= process_data(
        train_data,
        cat_features,
        num_features,
        discarded
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smote = SMOTE(random_state=42, sampling_strategy=0.4)
    train_X_resampled, train_y_resampled = smote.fit_resample(
        pd.concat([train_cat_X, train_num_X], axis=1), train_y['Is Fraud?']
    )
    # Resampled 데이터를 나누기
    train_cat_X_resampled = train_X_resampled[cat_features]
    train_num_X_resampled = train_X_resampled[num_features]
    train_y_resampled = pd.DataFrame(train_y_resampled, columns=['Is Fraud?'])

    # 데이터로더 초기화
    train_dataset = AE_validDataset(train_cat_X_resampled, train_num_X_resampled, train_y_resampled, device)
    valid_dataset = AE_validDataset(valid_cat_X, valid_num_X, valid_y, device)
    train_embeddings = model.get_embedding(
        torch.tensor(train_cat_X_resampled.values, dtype=torch.long).to(device),
        torch.tensor(train_num_X_resampled.values, dtype=torch.float).to(device),
    )
    valid_embeddings = model.get_embedding(
        torch.tensor(valid_cat_X.values, dtype=torch.long).to(device),
        torch.tensor(valid_num_X.values, dtype=torch.float).to(device),
    )
    train_embeddings = train_embeddings.cpu().detach().numpy()
    valid_embeddings = valid_embeddings.cpu().detach().numpy()
    rf_classifier = RandomForestClassifier(
        random_state=42,
        n_estimators=300,          # 트리 개수 더욱 증가
        max_depth=15,              # 더 깊은 트리 허용
        min_samples_leaf=1,        # 리프 노드 최소 샘플 수 더 감소
        min_samples_split=3,       # 분할 기준 완화
        class_weight={0: 1, 1: 12},  # 사기 클래스에 더 높은 가중치 부여
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,           # Out-of-bag 점수 확인
        n_jobs=-1
    )
    rf_classifier.fit(train_embeddings, train_y_resampled)
    y_pred = rf_classifier.predict(valid_embeddings)
    conf_matrix = confusion_matrix(valid_y, y_pred)
    class_report = classification_report(valid_y, y_pred)
    print(conf_matrix)
    print(class_report)
    #dump(rf_classifier, dt_filename)
    model_data = {
        "model": rf_classifier,
        "tree_le": le,
        "tree_scaler": scaler,
        "le": le_deep,
        "scaler": scaler_deep
    }
    dump(model_data, dt_filename)
    print("DT 저장 완료")


def TrainDAGMM(model, config, train_data):
    dagmm_filename = "dagmm.pth"
    cat_features, num_features, discarded = SelectFeature()
    (train_cat_X, train_num_X, train_y), (valid_cat_X, valid_num_X, valid_y), _ , _, _= process_data(
        train_data,
        cat_features,
        num_features,
        discarded
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 데이터로더 초기화
    train_dataset = AE_trainDataset(train_cat_X, train_num_X, device)
    valid_dataset = AE_validDataset(valid_cat_X, valid_num_X, valid_y, device)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='max',
        factor=0.2,
        patience=5,
        verbose=True,
        min_lr=1e-6,
    )

    best_f1 = 0

    for epoch in trange(config["epochs"], desc="Training", position=0, leave=True):
        model.train()
        train_loss = 0
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

        if epoch % 10 == 0:
            # 검증
            model.eval()
            valid_loss = 0
            recon_errors = []
            all_labels = []
            all_z = []

            with torch.no_grad():
                N = 0
                mu_sum = 0
                cov_sum = 0
                gamma_sum = 0

                for cat_features, num_features in tqdm(train_loader, desc="Valid Train", leave=False, position=1):
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

                for cat_features, num_features, labels in tqdm(valid_loader, desc="Valid Valid", leave=False, position=1):
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
                # 4. 스케줄러 업데이트 및 현재 학습률 로깅
                scheduler.step(f1)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                      f"Valid Loss = {valid_loss:.4f}, "
                      f"F1 Score = {f1:.4f}, "
                      f"LR = {current_lr:.6f}")
                    # 최고 성능 모델 저장 및 혼동 행렬 생성
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), dagmm_filename)
                    print(f"새로운 DAGMM 모델 저장됨 (F1 Score: {f1:.4f})")


def Train():
    config = {
        "encoding_dim": 28,
        "batch_size": 1024,
        "lr": 1e-4,
        "epochs": 300,
        "threshold_percentile": 90,
        "n_gmm": 8,
        "lambda_energy": 0.1,
        "lambda_cov_diag" : 0.005,
    }
    train_file = './data/[24-2 DS_Project2] Data.csv'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cat_features, num_features, discard = SelectFeature()

    tree_filename = "tree.pth"
    autoencoder_filename = "autoencoder.pth"

    config = {
        "encoding_dim": 31,
        "batch_size": 256,
        "lr": 1e-4,  # lr을 trial 파라미터로 변경
        "epochs": 200,
        "threshold_percentile": 95,
        "l1_lambda": 3e-6,
    }
    # train AutoEncoder
    ae = AutoEncoder(
        encoding_dim=config["encoding_dim"],
        cat_features=cat_features,
        num_features=num_features
    ).to(device)

    TrainAE(ae, config, train_file)

    # train encoding tree
    aeForDT = AutoEncoder(
        encoding_dim=config["encoding_dim"],
        cat_features=cat_features,
        num_features=num_features
    ).to(device)
    aeForDT.load_state_dict(torch.load(autoencoder_filename))

    TrainDT(aeForDT, config, train_file)

    # train dagmm
    config = {
        "encoding_dim": 28,
        "batch_size": 512,
        "lr": 1e-4,
        "epochs": 200,
        "threshold_percentile": 90,
        "n_gmm": 8,
        "lambda_energy": 0.1,
        "lambda_cov_diag" : 0.005,
    }
    dagmm = DAGMM(
        encoding_dim=config["encoding_dim"],
        n_gmm=config["n_gmm"],
        cat_features=cat_features,
        num_features=num_features,
    ).to(device)

    TrainDAGMM(dagmm, config, train_file)





if __name__ == '__main__':
    Train()
