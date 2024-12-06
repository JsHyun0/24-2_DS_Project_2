from sympy.codegen.cnodes import sizeof

from Models.model import BaseModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.mixture as mixture
from torch.autograd import Variable

class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.potrf(a, False)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s



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
            #layers += [nn.Tanh()]
            #nn.BatchNorm1d(48),
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
        #print(f"gamma Shape: {gamma.shape}")  # 예상: (N, K)

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
            #cov_inv = torch.linalg.inv(cov_k)
            #cov_inv.append(cov_inv.unsqueeze(0))
            cov_inv.append(torch.inverse(cov_k).unsqueeze(0))

            # Determinant of covariance
            det_cov.append(torch.det(cov_k).unsqueeze(0))
            #det_cov.append(np.linalg.det(cov_k.data.cpu().numpy()* (2*np.pi)))
            #det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))

            # Diagonal of covariance
            cov_diag += torch.sum(1 / cov_k.diagonal())

        cov_inv = torch.cat(cov_inv, dim=0)
        det_cov = torch.cat(det_cov).cuda()


        '''
        cov_inv = torch.linalg.inv(cov + torch.eye(cov.size(-1)).to(z.device) * eps)
        cov_inv = cov_inv.unsqueeze(0).repeat(z.size(0), 1, 1, 1)  # (128, 2, 29, 29)


        conv1 = (cov_inv @ z_mu.unsqueeze(-1))
        #print(f"conv1: {conv1.shape}")
        conv2 = z_mu.unsqueeze(-1) * conv1
        #print(f"conv2: {conv2.shape}")
        '''


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

        # GMM 매개변수 계산
        phi, mu, cov = self.compute_gmm_params(z, gamma)

        # 에너지 계산
        energy = self.compute_energy(z, phi, mu, cov)

        # 공분산 정규화
        cov_diag = torch.mean(torch.diagonal(cov, dim1=-2, dim2=-1))

        # 총 손실
        total_loss = recon_loss + lambda_energy * energy + lambda_cov_diag * cov_diag
        return total_loss, recon_loss, energy, cov_diag