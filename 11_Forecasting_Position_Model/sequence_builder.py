import numpy as np

class SequenceBuilder:

    def __init__(self, window_size):
        self.window_size = window_size

    ##### window 1: bars 1-100  --> predict bar 101
    ##### window 2: bars 2-101  --> predict bar 102
    def create_sequences(self, X, y):
        X_sequences = []
        y_sequences = []

        for i in range(self.window_size, len(X)):
            X_sequences.append(X[i - self.window_size:i])
            y_sequences.append(y[i])

        return (np.array(X_sequences), np.array(y_sequences))

    