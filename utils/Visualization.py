import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
    plt.show()
