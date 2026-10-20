from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from keras.models import Model
import numpy as np


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


def KerasSigmoidClassifier(X_train, y_train, X_test, y_test):

    n_features = X_train.shape[1]
    inputs = Input(shape=(n_features,))
    
    x = Dense(64)(inputs)
    x = Activation('sigmoid')
    x = Dropout(0.3)
    
    x = Dense(32)(x)
    x = Activation('sigmoid')
    
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    model.fit(
        X_train, y_train, 
        epochs=10, 
        batch_size=32, 
        validation_split=0.1
    )
    
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy with Sigmoid Hidden Layers: {accuracy:.4f} and the loss is {loss:.4f}")

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




def tensorflow_classifier(X_train, y_train, X_test, y_test, epochs=10, lr=0.001):
    
    X_train_tensor = tf.convert_to_tensor(X_train.toarray(), dtype=tf.float32)
    X_test_tensor = tf.convert_to_tensor(X_test.toarray(), dtype=tf.float32)
    
    y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
    
    num_features = X_train_tensor.shape[1]
    num_classes = y_train_tensor.shape[1]

    W1 = tf.Variable(tf.random.normal([num_features, 64], stddev=0.1))
    b1 = tf.Variable(tf.zeros([64]))

    W2 = tf.Variable(tf.random.normal([64, num_classes], stddev=0.1))
    b2 = tf.Variable(tf.zeros([num_classes]))

    def forward_pass(X):
        layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
        
        logits = tf.matmul(layer1, W2) + b2
        output = tf.nn.softmax(logits)
        return output

    optimizer = tf.optimizers.Adam(learning_rate=lr)

    print("Starting raw TensorFlow training...")
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            predictions = forward_pass(X_train_tensor)
            
            loss = tf.keras.losses.categorical_crossentropy(y_train_tensor, predictions)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, [W1, b1, W2, b2])

        optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2]))
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.numpy():.4f}")

    test_preds = forward_pass(X_test_tensor)
    
    predicted_classes = tf.argmax(test_preds, axis=1)
    true_classes = tf.argmax(y_test_tensor, axis=1)
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_classes, true_classes), tf.float32))
    print(f"Test Accuracy: {accuracy.numpy():.4f}")

    return W1, b1, W2, b2