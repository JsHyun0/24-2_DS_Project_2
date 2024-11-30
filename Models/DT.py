import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.decomposition import PCA, SparsePCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


# def DT(train_X, train_y, train_encoded, valid_X, valid_y, valid_encoded, random_state=42, max_depth=10, min_samples_leaf=5, class_weight={0: 1, 1: 10}) :
#     dt_classifier = DecisionTreeClassifier(
#         random_state=random_state,
#         max_depth=max_depth,
#         min_samples_leaf=min_samples_leaf,
#         class_weight=class_weight
#     )
#     dt_classifier.fit(train_X, train_y)
#
#     y_pred = dt_classifier.predict(valid_encoded)
#     conf_matrix = confusion_matrix(valid_y, y_pred)
#
#     print(confusion_matrix(valid_y, y_pred))
#     print(classification_report(valid_y, y_pred))

def showHeatMap(conf_matrix, le) :
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix-RandomForest')
    plt.show()

def weigthed_F1(conf_matrix):
    # 성능 지표 계산
    TP = conf_matrix[1, 1]  # True Positive
    TN = conf_matrix[0, 0]  # True Negative
    FP = conf_matrix[0, 1]  # False Positive = 오탐지
    FN = conf_matrix[1, 0]  # False Negative = 미탐지

    # 정확도, 정밀도, 재현율 계산
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # 가중치를 반영한 F1 점수 계산
    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def randomforest(#train_X,
                 train_y,
                 encoded_t,
                 encoded_v,
                 #valid_X,
                 valid_y,
                 le,
                 random_state=42,
                 max_depth=10,
                 min_samples_leaf=3,
                 class_weight={0: 1, 1: 10}) :
    rf_classifier = RandomForestClassifier(
        random_state=random_state,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight
    )
    rf_classifier.fit(encoded_t, train_y)

    y_pred = rf_classifier.predict(encoded_v)

    conf_matrix = confusion_matrix(valid_y, y_pred)

    showHeatMap(conf_matrix, le)

    print("report: ")
    print(classification_report(valid_y, y_pred))



