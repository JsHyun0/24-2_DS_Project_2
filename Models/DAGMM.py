from sympy.solvers.diophantine.diophantine import reconstruct

from Models.model import BaseModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.mixture as mixture


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
            nn.Linear(encoding_dim + 2, 10),
            #layers += [nn.Tanh()]
            #nn.BatchNorm1d(48),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(10, n_gmm),
            nn.Softmax(dim=1)
        )

        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm, encoding_dim + 2))
        self.register_buffer("cov", torch.zeros(n_gmm, encoding_dim + 2, encoding_dim + 2))

    def euclidian_distance(self, x, y):
        return (x-y).norm(2, dim=1) / x.norm(2, dim=1)

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)  # 샘플 개수
        sum_gamma = torch.sum(gamma, dim=0)

        # GMM 매개변수 업데이트
        phi = sum_gamma / N
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)
        cov = torch.sum(
            gamma.unsqueeze(-1).unsqueeze(-1) * (z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)),
            dim=0,
            ) / sum_gamma.unsqueeze(-1).unsqueeze(-1)

        self.phi = phi.data
        self.mu = mu.data
        self.cov = cov.data

        return phi, mu, cov

    def compute_energy(self, z, phi, mu, cov):
        eps = 1e-12
        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)
        cov_inv = torch.linalg.inv(cov + torch.eye(cov.size(-1)).to(z.device) * eps)

        exp_term = -0.5 * torch.sum(
            z_mu.unsqueeze(-1) * (cov_inv @ z_mu.unsqueeze(-2)), dim=(-1, -2)
        )
        log_term = torch.log(phi + eps) - 0.5 * torch.linalg.slogdet(cov + eps)[1]
        log_prob = exp_term + log_term

        energy = -torch.logsumexp(log_prob, dim=1)
        return energy

    # 임베딩 추출, torch.eval() 모드에서 사용
    def get_embedding(self, x_cat, x_num):
        with torch.no_grad():
            embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
            original_x = torch.cat(embeddings + [x_num], dim=1)
            x = self.fc_cat(original_x)
            encoded = self.encoder(x)
        return encoded

    # 학습과정
    def forward(self, x_cat, x_num):
        original_x = torch.cat([x_cat]+ [x_num], dim=1)
        embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        x = torch.cat(embeddings + [x_num], dim=1)

        x = self.fc_cat(x)
        enc = self.encoder(x)

        dec = self.AEDecoder(enc)


        rec_cosine = F.cosine_similarity(x, dec, dim=1)
        rec_euclidian = self.euclidian_distance(x, dec)

        z = torch.cat([enc, rec_euclidian.unsqeeze(-1), rec_cosine.unsqueeze(-1)], dim=1)

        gamma = self.estimation(z)

        return enc, dec, z, gamma

    def loss(self, x, x_hat, z, gamma):
        # 복원 손실 계산
        recon_loss = torch.mean((x - x_hat) ** 2)

        # GMM 매개변수 계산
        phi, mu, cov = self.compute_gmm_params(z, gamma)

        # 에너지 계산
        energy = torch.mean(self.compute_energy(z, phi, mu, cov))

        # 공분산 정규화
        cov_diag = torch.mean(torch.diagonal(cov, dim1=-2, dim2=-1))

        # 총 손실
        total_loss = recon_loss + self.lambda_energy * energy + self.lambda_cov_diag * cov_diag
        return total_loss, recon_loss, energy, cov_diag