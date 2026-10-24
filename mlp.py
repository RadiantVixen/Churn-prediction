import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

T = 1
Lr = 0.001
Epochs = 10


# def crossEntropy(prediction, label):
#     target = np.zeros_like(prediction)
#     target[int(label)] = 1.0
#     return prediction - target

def crossEntropy(prediction, label):
    return prediction - label

def cross_entropy_loss(probability, label):
    epsilon = 1e-12
    probability = np.clip(probability, epsilon, 1 - epsilon)
    return -(
        label * np.log(probability)
        + (1 - label) * np.log(1 - probability)
    )

def softMax(logits):
    logits = np.asarray(logits, dtype=float)
    shifted_logits = logits - np.max(logits)
    exponentials = np.exp(shifted_logits)
    probabilities = exponentials / np.sum(exponentials)

    return probabilities

def activation(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoidDerivative(a):
    return a * (1 - a)


def ft_gradient(delta, a, weights):
    return np.dot(weights.T, delta) * sigmoidDerivative(a)

class MLP_model:
    def __init__(self, l0):
        self.weights = [
        np.random.randn(64, l0) * np.sqrt(1.0 / l0),
        np.random.randn(32, 64) * np.sqrt(1.0 / 64),
        np.random.randn(16, 32) * np.sqrt(1.0 / 32),
        np.random.randn(1, 16) * np.sqrt(1.0 / 16),
        ]

        self.biases = [
        np.zeros(64),
        np.zeros(32),
        np.zeros(16),
        np.zeros(1),]

        self.layers = [np.zeros(64), np.zeros(32), np.zeros(16), np.zeros(1)]

    def backPropagation(self, error, feature):

        L4_gradient = error
        L3_gradient = ft_gradient(L4_gradient, self.layers[3],  self.weights[3])
        L2_gradient = ft_gradient(L3_gradient, self.layers[2],  self.weights[2])
        L1_gradient = ft_gradient(L2_gradient, self.layers[1], self.weights[1])

        self.biases[3] -= Lr * L4_gradient
        self.weights[3] -= Lr * np.outer(L4_gradient, self.layers[3])

        self.biases[2] -= Lr * L3_gradient
        self.weights[2] -= Lr * np.outer(L3_gradient, self.layers[2])
        
        self.biases[1] -= Lr * L2_gradient
        self.weights[1] -= Lr * np.outer(L2_gradient, self.layers[1])

        self.biases[0] -= Lr * L1_gradient
        self.weights[0] -= Lr * np.outer(L1_gradient, feature)
        


    def forwardPropagation(self, sample):

        z = np.dot(self.weights[0], sample) + self.biases[0]
        neural = activation(z)
        self.layers[1] = neural

        
        z = np.dot(self.weights[1], self.layers[1]) + self.biases[1]
        neural = activation(z)
        self.layers[2] = neural

        z = np.dot(self.weights[2], self.layers[2]) + self.biases[2]
        neural = activation(z)
        self.layers[3] = neural

        z = np.dot(self.weights[3], self.layers[3]) + self.biases[3]
        neural = activation(z)
        l5 = neural

        return l5


    def train(self, features, lables):

        for epoch in range(Epochs):
            indices = np.arange(len(features))
            np.random.shuffle(indices)

            for i in indices:
                prediction = self.forwardPropagation(features[i])
                error = crossEntropy(prediction, lables[i])
                self.backPropagation(error, features[i])



def predict(ids, X_train, X_test, y_train, y_test):

    correct = 0
    customer_probabilities = []

    model = MLP_model(X_train.shape[1])
    model.train(X_train, y_train)


    for i in range(len(X_test)):
        prob = model.forwardPropagation(X_test[i])
        customer_probabilities.append(prob)

        if prob >= 0.5:
            prediction = 1
        else:
            prediction = 0

        if prediction == y_test[i]:
            correct += 1

    # loss = cross_entropy_loss(np.array(customer_probabilities), y_test)
    accuracy = correct / len(y_test) * 100

    # print(f"Binary cross-entropy: {loss.mean():.4f}")
    print(f"\nAccuracy: {accuracy:.2f}%")
    print(f"Correct predictions: {correct}/{len(y_test)}")


    probs_array = np.array(customer_probabilities)
    binary_predictions = (probs_array >= 0.5).astype(int)

    # pd.DataFrame({"ID": ids, "TARGET": binary_predictions}).to_csv(
    #     "customer_probabilities.csv", index=False
    # )

    zeros_count = np.sum(y_train == 0)
    ones_count = np.sum(y_train == 1)

    print(f"y train Zeros (0): {zeros_count}")
    print(f"y tain Ones (1): {ones_count}")

    print("Min:", np.min(customer_probabilities))
    print("Max:", np.max(customer_probabilities))
    print("Mean:", np.mean(customer_probabilities))
    print("Std:", np.std(customer_probabilities))
    print("Unique predictions:", np.unique(customer_probabilities).size)

    metrics = {
        "accuracy": accuracy_score(y_test, binary_predictions),
        "auc": roc_auc_score(y_test, probs_array)
    }
    
    return metrics
