import json
import numpy as np

T = 1
L1_weights = np.random.randn(50, 102) * np.sqrt(1.0 / 102)
L2_weights = np.random.randn(25, 50) * np.sqrt(1.0 / 50)
L3_weights = np.random.randn(2, 25) * np.sqrt(1.0 / 25)
Lr = 0.001
Epochs = 1

L1_biases = np.zeros(50)
L2_biases = np.zeros(25)
L3_biases = np.zeros(2)

l2 = [0] * 50
l3 = [0] * 25
l4 = [0] * 2

def crossEntropy(prediction, label):
    return np.array([prediction[0] - label, prediction[1] - (1 - label)])


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
    print("np.dot(weights.T, delta)",np.dot(weights.T, delta).shape, "weights.shape", weights.shape, "delta", delta.shape, "sigmoidDerivative(a)", sigmoidDerivative(a).shape)
    return np.dot(weights.T, delta) * sigmoidDerivative(a).reshape(-1, 1)


def backPropagation(error, feature):
    global L1_weights, L2_weights, L3_weights, L1_biases, L2_biases, L3_biases

    L3_gradient = error
    L2_gradient = ft_gradient(L3_gradient, l3,  L3_weights)
    L1_gradient = ft_gradient(L2_gradient, l2, L2_weights)


    L3_biases -= Lr * L3_gradient.reshape(-1)
    L3_weights -= Lr * np.outer(L3_gradient, l3)
    
    L2_biases -= Lr * L2_gradient.reshape(-1)
    L2_weights -= Lr * np.outer(L2_gradient, l2)

    L1_biases -= Lr * L1_gradient.reshape(-1)
    L1_weights -= Lr * np.outer(L1_gradient, feature)


def forwardPropagation(sample):
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


def cross_entropy_loss(probability_m, label):
    epsilon = 1e-12
    probability_m = np.clip(probability_m, epsilon, 1 - epsilon)
    return -(
        label * np.log(probability_m)
        + (1 - label) * np.log(1 - probability_m)
    )



def train(features, lables):

    for epoch in range(Epochs):
        indices = np.arange(len(features))
        np.random.shuffle(indices)

        for i in indices:
            prediction = forwardPropagation(features[i])
            error = crossEntropy(prediction, lables.iloc[i])
            backPropagation(error, features[i])



def predict(ids, X_train, X_test, y_train, y_test):

    correct = 0
    malignant_probabilities = []

    train(X_train[:100], y_train)


    for i in range(len(X_test)):
        M, B = forwardPropagation(X_test[i])
        malignant_probabilities.append(M)

        if M > B:
            prediction = 1
        else:
            prediction = 0

        # print(ids[i], end=": ")

        if prediction == 1:
            print("M")
        else:
            print("B")

        if prediction == y_test[i]:
            correct += 1

    loss = cross_entropy_loss(np.array(malignant_probabilities), y_test)
    accuracy = correct / len(y_test) * 100

    print(f"Binary cross-entropy: {loss:.4f}")
    print(f"\nAccuracy: {accuracy:.2f}%")
    print(f"Correct predictions: {correct}/{len(y_test)}")

