import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Bidirectional, Input, Dropout)

class BiLSTMModel:

    def __init__(self, window_size, n_features):
        self.window_size    = window_size
        self.n_features     = n_features

        self.model      = self.build_model()

    def build_model(self):
        model = Sequential([Input(shape=(self.window_size, self.n_features)), Bidirectional(LSTM(64)), Dropout(0.2), Dense(32, activation="relu"),Dense(1)])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss="mae", metrics=["mae"])

        return model

    def fit(self, X_train, y_train, epochs=30, batch_size=32, validation_data=None):
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, shuffle=False)

    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()

    def save(self, path):
        self.model.save(path)

        

    