from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class NumPyMLP:
    def __init__(self, layer_sizes, learning_rate=0.005, epochs=90, batch_size=64):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        

        self.weights = []
        self.biases = []
        
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(0.001 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return (z > 0).astype(float)

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        num_samples = X.shape[0]

        for epoch in range(self.epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]


            for i in range(0, num_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                activations = [X_batch]
                zs = []
                
                current_input = X_batch
                for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
                    z = np.dot(current_input, w) + b
                    zs.append(z)
                    
                    if idx == len(self.weights) - 1:
                        activation = self._sigmoid(z)
                    else:
                        activation = self._relu(z)
                        
                    activations.append(activation)
                    current_input = activation

                pos_weight = 3
                sample_weights = np.where(y_batch == 1, pos_weight, 1.0)
                
                output_error = (activations[-1] - y_batch) * sample_weights
                deltas = [output_error]

                for idx in range(len(self.weights) - 2, -1, -1):
                    error = np.dot(deltas[0], self.weights[idx+1].T)
                    delta = error * self._relu_derivative(zs[idx])
                    deltas.insert(0, delta)

                for idx in range(len(self.weights)):
                    grad_w = np.dot(activations[idx].T, deltas[idx]) / X_batch.shape[0]
                    grad_b = np.sum(deltas[idx], axis=0, keepdims=True) / X_batch.shape[0]

                    self.weights[idx] -= self.lr * grad_w
                    self.biases[idx] -= self.lr * grad_b

    def predict_proba(self, X):
        current_input = X
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(current_input, w) + b
            if idx == len(self.weights) - 1:
                current_input = self._sigmoid(z)
            else:
                current_input = self._relu(z)
        return current_input.ravel()



def evaluate_numpy_mlp(X_train, y_train, X_test, y_test):
    imputer = SimpleImputer(strategy='median')
    X_train_clean = imputer.fit_transform(X_train)
    X_test_clean = imputer.transform(X_test)


    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)


    selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    selector.fit(X_train_scaled, y_train)

    importances = selector.feature_importances_
    top_indices = np.argsort(importances)[-50:]

    X_train_filtered = X_train_scaled[:, top_indices]
    X_test_filtered = X_test_scaled[:, top_indices]
    input_dim = X_train_filtered.shape[1]
    model = NumPyMLP(layer_sizes=[input_dim, 64, 32, 1], learning_rate=0.005, epochs=90, batch_size=64)
    model.fit(X_train_filtered, y_train)

    probabilities = model.predict_proba(X_test_filtered)
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "auc": roc_auc_score(y_test, probabilities)
    }
    
    print(f"NumPy MLP Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"NumPy MLP Test AUC: {metrics['auc']:.4f}")
    

    # pd.DataFrame({"ID": test_df["ID"], "TARGET": predictions}).to_csv("predictions.csv", index=False)

    # fpr, tpr, _ = roc_curve(y_test, probabilities)
    # roc_auc = auc(fpr, tpr)
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
    # plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend()
    # plt.show()

    
    return model, metrics