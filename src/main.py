import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        y_pred_prob = self.predict_proba(X)
        return np.where(y_pred_prob >= 0.5, 1, 0)

if __name__ == "__main__":

    plots_dir = "../Plots"
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv("../Data/proc_pima_2_withheader.csv_preprocessed.csv")

    X = df.drop("Diabetes", axis=1).values
    y = df["Diabetes"].replace(-1, 0).values   

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


    scratch_model = LogisticRegressionScratch(lr=0.01, epochs=5000)
    scratch_model.fit(X_train, y_train)
    y_pred_scratch = scratch_model.predict(X_test)

    acc_scratch = accuracy_score(y_test, y_pred_scratch)
    prec_scratch = precision_score(y_test, y_pred_scratch)
    rec_scratch = recall_score(y_test, y_pred_scratch)
    f1_scratch = f1_score(y_test, y_pred_scratch)

    print("\n[Scratch Logistic Regression]")
    print(f"Accuracy: {acc_scratch:.4f}")
    print(f"Precision: {prec_scratch:.4f}")
    print(f"Recall: {rec_scratch:.4f}")
    print(f"F1 Score: {f1_scratch:.4f}")

    
    sklearn_model = LogisticRegression(max_iter=1000)
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)

    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    prec_sklearn = precision_score(y_test, y_pred_sklearn)
    rec_sklearn = recall_score(y_test, y_pred_sklearn)
    f1_sklearn = f1_score(y_test, y_pred_sklearn)

    print("\n[Sklearn Logistic Regression]")
    print(f"Accuracy: {acc_sklearn:.4f}")
    print(f"Precision: {prec_sklearn:.4f}")
    print(f"Recall: {rec_sklearn:.4f}")
    print(f"F1 Score: {f1_sklearn:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_sklearn)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Sklearn Model)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
    plt.close()

    # ROC Curve
    y_proba_scratch = scratch_model.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_proba_scratch)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"Scratch AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Scratch Model)")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "roc_curve.png"))
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba_scratch)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, marker='.')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Scratch Model)")
    plt.savefig(os.path.join(plots_dir, "precision_recall_curve.png"))
    plt.close()

    print(f"\n Plots saved in: {plots_dir}")
