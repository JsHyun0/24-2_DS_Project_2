from sklearn.mixture import GaussianMixture

# GMM for DAGMM
class GMM:
    """
    GMM 초기화
    Args:
        n_components: 클러스터 수
            정상 데이터 분포를 의미
        covariance_type: 공분산 매트릭스 타입

    """
    def __init__(self, n_components, covariance_type='full', random_state=42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.gmm = None

    """
    데이터 학습
    Args: features: 학습할 데이터(numpy.ndarray, 샘플 x 특징 벡터), 
        샘플 x (Latent vector(encoded_dim) + (reconstruction error)1)
    Returns: self: 학습된 gmm 객체 반환
    """
    def train(self, features):
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state
        )
        self.gmm.fit(features) # GMM 학습
        return self

