import tensorflow as tf
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Input, Dense, concatenate
from keras.models import Model

def baseline(X_train, y_train, X_test, y_test):

    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    y_pred = dummy_clf.predict(X_test)
    dummy_clf.score(y_test, y_pred)
    return dummy_clf

def randomForest(X_train, y_train, X_test, y_test):

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    grid_search.score(y_test, y_pred)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Estimator:", grid_search.best_estimator_)
    return grid_search

def mlpClassifier(X_train, y_train, X_test, y_test):

    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    clf.predict_proba(X_test[:1])
    y_pred = clf.predict(X_test[:5, :])
    clf.score(y_test, y_pred)
    return clf

def KerasClassifier(X_train, y_train, X_test, y_test):

    model = Sequential()
    model.add(Dense(units=64, input_dim=100))
    model.add(Activation('relu'))
    model.add(Dense(units=10))
    model.add(Activation('softmax'))
    input1 = Input(shape=(100,))
    input2 = Input(shape=(50,))
    hidden1 = Dense(64, activation='relu')(input1)
    hidden2 = Dense(32, activation='relu')(input2)
    merged = concatenate([hidden1, hidden2])
    output = Dense(10, activation='softmax')(merged)
    model = Model(inputs=[input1, input2], outputs=output)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([X_train, X_train], y_train, epochs=10, batch_size=32)
    y_pred = model.predict([X_test, X_test])
    model.evaluate([X_test, X_test], y_test)


    return model

def tanserflowClassifier(X_train, y_train, X_test, y_test):
    def forward_pass(X):
        layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
        output = tf.sigmoid(tf.matmul(layer1, W2) + b2)
        return output

    W1 = tf.Variable(tf.random.normal([num_features, 64], stddev=0.1))
    b1 = tf.Variable(tf.zeros([64]))

    W2 = tf.Variable(tf.random.normal([64, 1], stddev=0.1))
    b2 = tf.Variable(tf.zeros([1]))

    optimizer = tf.optimizers.Adam(learning_rate=0.001)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            predictions = forward_pass(X_train_tensor)
            loss = tf.keras.losses.binary_crossentropy(y_train_tensor, predictions)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, [W1, b1, W2, b2])

        optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2]))


