import numpy as np

class Perceptron:
    def __init__(self, num_epochs=10):
        self.num_epochs = num_epochs
        self.w = None
        self.b = 0

    def train(self, X, y):
        """
        TODO: Implement the Perceptron Update Rule.
        1. Init w and b to zeros (w is a vector and b is a scalar).
        2. Loop epochs.
        3. Loop examples:
           If prediction is wrong:
              w = w + y * x
              b = b + y
        """
        pass

    def predict(self, X):
        """
        TODO: Compute w*x + b. Return +1 or -1.
        """
        return np.zeros(X.shape[0])