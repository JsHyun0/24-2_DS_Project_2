import optuna
import neptune
import json
from experiments import autoencoder_experiment_ver4_1

if __name__ == "__main__":

    
    # Optuna 학습 시작 (최대화 방향으로 변경)
    study = optuna.create_study(direction="maximize")
    study.optimize(autoencoder_experiment_ver4_1.objective, n_trials=40)
    
    # 최적의 하이퍼파라미터 저장
    with open("experiments/best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)