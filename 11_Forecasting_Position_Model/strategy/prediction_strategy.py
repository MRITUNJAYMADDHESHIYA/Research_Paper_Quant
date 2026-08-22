class PredictionStrategy:

    def __init__(
        self,
        threshold=0.001
    ):

        self.threshold = threshold

    def generate_signal(
        self,
        predicted_return
    ):

        if predicted_return > self.threshold:

            return 1       # BUY

        elif predicted_return < -self.threshold:

            return -1      # SELL

        else:

            return 0       # HOLD