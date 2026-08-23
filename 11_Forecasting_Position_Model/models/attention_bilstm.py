import tensorflow as tf
from tensorflow.keras.layers import (Layer, Input, LSTM, Dense, Dropout, Bidirectional)
from tensorflow.keras.models import Model


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1], 1), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="attention_bias",    shape=(input_shape[1], 1), initializer="zeros",          trainable=True)
        super().build(input_shape)

    def call(self, x):
        ####### Attention score
        score = tf.tanh(tf.matmul(x, self.W) + self.b)

        ####### Attention weights
        attention_weights = tf.nn.softmax(score,axis=1)

        ####### Weighted hidden states
        context = (x * attention_weights)
        context = tf.reduce_sum(context, axis=1)

        return context


class AttentionBiLSTMModel:
    def __init__(self,window_size,n_features):
        self.window_size = window_size
        self.n_features  = n_features
        self.model       = self.build_model()

    def build_model(self):
        inputs = Input(shape=(self.window_size, self.n_features))

        x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        x = AttentionLayer()(x)
        x = Dropout(0.2)(x)
        x = Dense(32,activation="relu")(x)

        outputs = Dense(1)(x)
        model   = Model(inputs=inputs,outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mae", metrics=["mae"])

        return model

    def fit(self, X_train, y_train, epochs=30, batch_size=32, validation_data=None):
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, shuffle=False)

    def predict(self, X):
        return self.model.predict(X,verbose=0).flatten()

    def save(self, path):
        self.model.save(path)

        
