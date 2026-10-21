import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier


def _metrics(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    return {"accuracy": accuracy_score(y_test, probabilities >= 0.5),
            "auc": roc_auc_score(y_test, probabilities)}


def baseline(X_train, y_train, X_test, y_test):
    model = DummyClassifier(strategy="most_frequent", random_state=42).fit(X_train, y_train)
    return model, _metrics(model, X_test, y_test)


def random_forest(X_train, y_train, X_test, y_test):
    model = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
        {"n_estimators": [100], "max_depth": [None, 12], "min_samples_leaf": [1, 3]},
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring="roc_auc", n_jobs=-1,
    ).fit(X_train, y_train)
    return model, _metrics(model, X_test, y_test)


def mlp_classifier(X_train, y_train, X_test, y_test):
    model = MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                          max_iter=300, random_state=42).fit(X_train, y_train)
    return model, _metrics(model, X_test, y_test)


def keras_classifier(X_train, y_train, X_test, y_test, epochs=20):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)), tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3), tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    probabilities = model.predict(X_test, verbose=0).ravel()
    return model, {"accuracy": accuracy_score(y_test, probabilities >= 0.5),
                   "auc": roc_auc_score(y_test, probabilities)}


def tensorflow_classifier(X_train, y_train, X_test, y_test, epochs=20, lr=0.001):
    import tensorflow as tf
    tf.random.set_seed(42)
    weights = [tf.Variable(tf.random.normal((X_train.shape[1], 64), stddev=0.1)), tf.Variable(tf.zeros(64)),
               tf.Variable(tf.random.normal((64, 1), stddev=0.1)), tf.Variable(tf.zeros(1))]
    optimizer = tf.keras.optimizers.Adam(lr)
    X_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y_train.reshape(-1, 1), dtype=tf.float32)
    for _ in range(epochs):
        with tf.GradientTape() as tape:
            hidden = tf.nn.relu(tf.matmul(X_tensor, weights[0]) + weights[1])
            logits = tf.matmul(hidden, weights[2]) + weights[3]
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_tensor, logits))
        optimizer.apply_gradients(zip(tape.gradient(loss, weights), weights))
    hidden = tf.nn.relu(tf.matmul(tf.convert_to_tensor(X_test, tf.float32), weights[0]) + weights[1])
    probabilities = tf.sigmoid(tf.matmul(hidden, weights[2]) + weights[3]).numpy().ravel()
    return weights, {"accuracy": accuracy_score(y_test, probabilities >= 0.5),
                     "auc": roc_auc_score(y_test, probabilities)}