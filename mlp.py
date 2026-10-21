import json
import numpy as np

T = 1
L1_weights = np.random.randn(50, 102) * np.sqrt(1.0 / 102)
L2_weights = np.random.randn(25, 50) * np.sqrt(1.0 / 50)
L3_weights = np.random.randn(2, 25) * np.sqrt(1.0 / 25)
Epochs = 1


class NumpyMLP:
    """A small OOP multilayer perceptron implemented with matrix operations."""

    def __init__(self, input_size, hidden_sizes=(64, 32), learning_rate=0.001, seed=42):
        rng = np.random.default_rng(seed)
        sizes = (input_size, *hidden_sizes, 1)
        self.weights = [rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1]))
                         for i in range(len(sizes) - 1)]
        self.biases = [np.zeros(size) for size in sizes[1:]]
        self.learning_rate = learning_rate

    @staticmethod
    def _sigmoid(values):
        return 1 / (1 + np.exp(-np.clip(values, -500, 500)))

    def predict_proba(self, features):
        activation = np.asarray(features, dtype=float)
        for weights, bias in zip(self.weights[:-1], self.biases[:-1]):
            activation = np.maximum(activation @ weights + bias, 0)
        return self._sigmoid(activation @ self.weights[-1] + self.biases[-1]).ravel()

    def fit(self, features, labels, epochs=100):
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels, dtype=float).reshape(-1, 1)
        for _ in range(epochs):
            activations = [features]
            for weights, bias in zip(self.weights[:-1], self.biases[:-1]):
                activations.append(np.maximum(activations[-1] @ weights + bias, 0))
            probabilities = self._sigmoid(activations[-1] @ self.weights[-1] + self.biases[-1])
            delta = probabilities - labels
            for layer in range(len(self.weights) - 1, -1, -1):
                self.weights[layer] -= self.learning_rate * (activations[layer].T @ delta) / len(features)
                self.biases[layer] -= self.learning_rate * delta.mean(axis=0)
                if layer:
                    delta = (delta @ self.weights[layer].T) * (activations[layer] > 0)
        return self


def predict(ids, X_train, X_test, y_train, y_test, epochs=100):
    from sklearn.metrics import accuracy_score, roc_auc_score
    model = NumpyMLP(X_train.shape[1]).fit(X_train, y_train, epochs)
    probabilities = model.predict_proba(X_test)
    metrics = {"accuracy": accuracy_score(y_test, probabilities >= 0.5),
               "auc": roc_auc_score(y_test, probabilities)}
    print(f"NumPy MLP: accuracy={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")
    return model, metrics
    L2_biases -= Lr * L2_gradient.reshape(-1)

    L2_weights -= Lr * np.outer(L2_gradient, l2)
