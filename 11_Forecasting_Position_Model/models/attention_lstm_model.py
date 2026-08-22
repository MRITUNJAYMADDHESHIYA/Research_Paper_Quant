import tensorflow as tf

from tensorflow import (Input, LSTM, Dense, Bidirectional)
from tensorflow import Model



class AttentionLSTMModel(BaseModel):

    def __init__(self, seq_len, n_features):
        self.seq_len    = seq_len
        self.n_features = n_features

        self.model      = self.build_model()

    def build_model(self):
        inputs = Input(shape=(self.seq_len, self.n_features))

        x = LSTM(64, return_sequences=True)(inputs)
        x = AttentionLayer()(x)
        outputs = Dense(1)(x)

        # x = Bidirectional(LSTM(64, return_sequence=True))(inputs)
        # x = AttentionLayer()(x)
        # x = Dense(32, activation="relu")(x)
        # outputs = Dense(1)(x)

        model = Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mae")

        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=20, batch_size=32)

    def predict(self, X):
        return self.model.predict(X)
    