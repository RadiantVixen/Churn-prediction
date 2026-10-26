import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class NumPyMLP:
    def __init__(self, layer_sizes, learning_rate=0.01, epochs=100, batch_size=32):
        self.layer_sizes = layer_sizes  # e.g., [input_dim, 64, 32, 1]
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Initialize weights and biases using He/Xavier initialization
        self.weights = []
        self.biases = []
        
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU layers
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return (z > 0).astype(float)

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        # Ensure y is a 2D column vector
        y = y.reshape(-1, 1)
        num_samples = X.shape[0]

        for epoch in range(self.epochs):
            # Shuffle dataset each epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch gradient descent
            for i in range(0, num_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                # --- 1. Forward Pass ---
                activations = [X_batch]
                zs = []
                
                current_input = X_batch
                for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
                    z = np.dot(current_input, w) + b
                    zs.append(z)
                    
                    # Hidden layers use ReLU, final output layer uses Sigmoid
                    if idx == len(self.weights) - 1:
                        activation = self._sigmoid(z)
                    else:
                        activation = self._relu(z)
                        
                    activations.append(activation)
                    current_input = activation

                # --- 2. Backward Pass ---
                # Derivative of Binary Cross Entropy with Sigmoid output simplifies to (y_pred - y_true)
                pos_weight = 3  # Amplifies the penalty for missing churners
                sample_weights = np.where(y_batch == 1, pos_weight, 1.0)
                
                output_error = (activations[-1] - y_batch) * sample_weights
                deltas = [output_error]

                # Backpropagate through hidden layers
                for idx in range(len(self.weights) - 2, -1, -1):
                    error = np.dot(deltas[0], self.weights[idx+1].T)
                    delta = error * self._relu_derivative(zs[idx])
                    deltas.insert(0, delta)

                # --- 3. Gradient Update ---
                for idx in range(len(self.weights)):
                    grad_w = np.dot(activations[idx].T, deltas[idx]) / X_batch.shape[0]
                    grad_b = np.sum(deltas[idx], axis=0, keepdims=True) / X_batch.shape[0]

                    # Apply gradient descent update
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
    # 1. Impute missing values using median (essential for 100+ feature financial datasets)
    imputer = SimpleImputer(strategy='median')
    X_train_clean = imputer.fit_transform(X_train)
    X_test_clean = imputer.transform(X_test)

    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    # 3. Initialize and train MLP
    # Architecture: Input -> 64 -> 32 -> 1 Output
    from sklearn.ensemble import RandomForestClassifier

    # Quick feature selector
    selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    selector.fit(X_train_scaled, y_train)

    # Keep top features based on importance
    importances = selector.feature_importances_
    top_indices = np.argsort(importances)[-50:]  # Keep top 50 features

    X_train_filtered = X_train_scaled[:, top_indices]
    X_test_filtered = X_test_scaled[:, top_indices]
    input_dim = X_train_filtered.shape[1]
    model = NumPyMLP(layer_sizes=[input_dim, 64, 32, 1], learning_rate=0.005, epochs=90, batch_size=64)
    model.fit(X_train_filtered, y_train)

    # 4. Generate continuous probabilities for AUC calculation
    probabilities = model.predict_proba(X_test_filtered)
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "auc": roc_auc_score(y_test, probabilities)
    }
    
    print(f"NumPy MLP Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"NumPy MLP Test AUC: {metrics['auc']:.4f}")

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    
    return model, metrics