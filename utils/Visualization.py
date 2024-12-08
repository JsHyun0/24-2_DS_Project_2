import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import re
import os

def plot_confusion_heatmap(y_true, y_pred, class_names=None, title="Confusion Matrix", cmap="Blues"):
    """
    예측 결과와 실제 값을 기반으로 혼동 행렬 히트맵을 생성하고 시각화합니다.

    Parameters:
        y_true (array-like): 실제 클래스 레이블.
        y_pred (array-like): 예측 클래스 레이블.
        class_names (list, optional): 클래스 이름. None일 경우 숫자 레이블 사용.
        title (str, optional): 히트맵의 제목.
        cmap (str, optional): 히트맵의 색상 맵.
    """
    # 혼동 행렬 계산
    cm = confusion_matrix(y_true, y_pred)

    # 클래스 이름이 없으면 숫자 레이블 사용
    if class_names is None:
        class_names = [str(i) for i in range(len(set(y_true)))]

    # 히트맵 생성
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.savefig("./temp.png", dpi=300, bbox_inches='tight')
    plt.show()

def visualize_loss(train_loss_path: str, valid_loss_path: str) :
    # CSV 파일 읽기
    # 컬럼명이 없으므로 names 파라미터를 사용하여 컬럼명 지정
    train_loss = pd.read_csv(train_loss_path, names=['index', 'count', 'loss'])
    valid_loss = pd.read_csv(valid_loss_path, names=['index', 'count', 'loss'])

    # 그래프 스타일 설정
    plt.figure(figsize=(12, 6))

    # loss 그래프 그리기
    plt.plot(train_loss['index'], train_loss['loss'], label='Train Loss', color='blue')
    plt.plot(valid_loss['index'], valid_loss['loss'], label='Validation Loss', color='red')

    # 그래프 꾸미기
    plt.title('Training and Validation Loss', fontsize=14, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    # x축 틱 간격 설정 (데이터에 따라 조절 가능)
    plt.xticks(rotation=45)

    # 그래프 여백 조정
    plt.tight_layout()

    number_match = re.search(r'train_loss_(\d+)\.csv', train_loss_path)
    if number_match: number = number_match.group(1)
    else : number = "unkown"

    save_dir = os.path.join('Artifacts', 'loss')

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f'loss_graph_{number}.png')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    # Example
    # visualize_loss(train_loss_path, valid_loss_path)