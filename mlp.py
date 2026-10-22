import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

T = 1
L1_weights = np.random.randn(64, 101) * np.sqrt(1.0 / 101)
L2_weights = np.random.randn(32, 64) * np.sqrt(1.0 / 64)
L3_weights = np.random.randn(2, 32) * np.sqrt(1.0 / 32)
Lr = 0.001
Epochs = 10

L1_biases = np.zeros(64)
L2_biases = np.zeros(32)
L3_biases = np.zeros(2)

l2 = [0] * 64
l3 = [0] * 32
l4 = [0] * 2

def crossEntropy(prediction, label):
    target = np.zeros_like(prediction)
    target[int(label)] = 1.0
    return prediction - target

def cross_entropy_loss(probability_m, label):
    epsilon = 1e-12
    probability_m = np.clip(probability_m, epsilon, 1 - epsilon)
    return -(
        label * np.log(probability_m)
        + (1 - label) * np.log(1 - probability_m)
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
    def __init__(self):
        print("init")

    def backPropagation(self, error, feature):
        global L1_weights, L2_weights, L3_weights, L1_biases, L2_biases, L3_biases

        L3_gradient = error
        L2_gradient = ft_gradient(L3_gradient, l3,  L3_weights)
        L1_gradient = ft_gradient(L2_gradient, l2, L2_weights)


        L3_biases -= Lr * L3_gradient
        L3_weights -= Lr * np.outer(L3_gradient, l3)
        
        L2_biases -= Lr * L2_gradient
        L2_weights -= Lr * np.outer(L2_gradient, l2)

        L1_biases -= Lr * L1_gradient
        L1_weights -= Lr * np.outer(L1_gradient, feature)
        


    def forwardPropagation(self, sample):
        global l2, l3, l4

        z = np.dot(L1_weights, sample) + L1_biases
        neural = activation(z)
        l2 = neural

        
        z = np.dot(L2_weights, l2) + L2_biases
        neural = activation(z)
        l3 = neural

        z = np.dot(L3_weights, l3) + L3_biases
        l4 = z

        return softMax(l4)


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

    model = MLP_model()
    model.train(X_train, y_train)

    for i in range(len(X_test)):
        probs = model.forwardPropagation(X_test[i])
        S, L = probs[0], probs[1]
        customer_probabilities.append(L)

        if S > L:
            prediction = 0
        else:
            prediction = 1

        if prediction == y_test[i]:
            correct += 1

    loss = cross_entropy_loss(np.array(customer_probabilities), y_test)
    accuracy = correct / len(y_test) * 100

    print(f"Binary cross-entropy: {loss.mean():.4f}")
    print(f"\nAccuracy: {accuracy:.2f}%")
    print(f"Correct predictions: {correct}/{len(y_test)}")


    probs_array = np.array(customer_probabilities)
    binary_predictions = (probs_array >= 0.5).astype(bool)

    pd.DataFrame({"ID": ids, "TARGET": binary_predictions}).to_csv(
        "customer_probabilities.csv", index=False
    )

    metrics = {
        "accuracy": accuracy_score(y_test, binary_predictions),
        "auc": roc_auc_score(y_test, probs_array)
    }
    
    return metrics
