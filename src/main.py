import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import os

# Set interactive Matplotlib backend for VS Code
plt.switch_backend('Qt5Agg')  

plots_dir = "../Plots"
os.makedirs(plots_dir, exist_ok=True)

data_path = "../Data/retail_sales_dataset_preprocessed.csv"
df = pd.read_csv(data_path)

print("Data Summary:")
print(df.describe())

X = df.drop(["Product_Category_Encoded", "Qty_Price_Interaction"], axis=1).values
y = df["Product_Category_Encoded"].values

if np.any(np.isnan(X)) or np.any(np.isnan(y)):
    print("Warning: NaN values found in data")
if np.any(np.isinf(X)) or np.any(np.isinf(y)):
    print("Warning: Infinite values found in data")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression from Scratch (Multi-class with Softmax)
class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.001, epochs=1000, clip_value=1.0):
        self.lr = learning_rate
        self.epochs = epochs
        self.clip_value = clip_value
        self.weights = None
        self.bias = None
        self.loss_history = []

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)

        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y.astype(int)] = 1

        # Gradient descent
        for epoch in range(self.epochs):
            # Forward pass
            logits = np.dot(X, self.weights) + self.bias
            y_pred = self.softmax(logits)

            # Compute gradients
            error = y_pred - y_one_hot
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error, axis=0)

            # Gradient clipping
            dw = np.clip(dw, -self.clip_value, self.clip_value)
            db = np.clip(db, -self.clip_value, self.clip_value)

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Compute cross-entropy loss
            loss = -np.mean(np.sum(y_one_hot * np.log(y_pred + 1e-10), axis=1))
            self.loss_history.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        logits = np.dot(X, self.weights) + self.bias
        y_pred = self.softmax(logits)
        return np.argmax(y_pred, axis=1)

# Train from-scratch model
model_scratch = LogisticRegressionFromScratch(learning_rate=0.001, epochs=1000, clip_value=1.0)
model_scratch.fit(X_train, y_train)
y_pred_scratch = model_scratch.predict(X_test)

# Train scikit-learn model
model_sklearn = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
model_sklearn.fit(X_train, y_train)
y_pred_sklearn = model_sklearn.predict(X_test)

# Calculating metrics 
metrics = {
    "Accuracy": (accuracy_score(y_test, y_pred_scratch), accuracy_score(y_test, y_pred_sklearn)),
    "Precision": (precision_score(y_test, y_pred_scratch, average='macro'), precision_score(y_test, y_pred_sklearn, average='macro')),
    "Recall": (recall_score(y_test, y_pred_scratch, average='macro'), recall_score(y_test, y_pred_sklearn, average='macro')),
    "F1 Score": (f1_score(y_test, y_pred_scratch, average='macro'), f1_score(y_test, y_pred_sklearn, average='macro'))
}


print("\nMetrics Comparison:")
print(f"{'Metric':<20} {'From Scratch':<15} {'Scikit-Learn':<15}")
print("-" * 50)
for metric, (scratch, sklearn) in metrics.items():
    print(f"{metric:<20} {scratch:.4f} {'':<5} {sklearn:.4f}")

# Plot Confusion Matrices
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Beauty', 'Clothing', 'Electronics'],
                yticklabels=['Beauty', 'Clothing', 'Electronics'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(plots_dir, filename))
    plt.show()

plot_confusion_matrix(y_test, y_pred_scratch, 
                      "Confusion Matrix (From Scratch)", 
                      "confusion_matrix_scratch.png")

plot_confusion_matrix(y_test, y_pred_sklearn, 
                      "Confusion Matrix (Scikit-Learn)", 
                      "confusion_matrix_sklearn.png")

# Plot 3: Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(model_scratch.loss_history)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training Loss Curve (From Scratch)")
plt.savefig(os.path.join(plots_dir, "loss_curve_scratch.png"))
plt.show()