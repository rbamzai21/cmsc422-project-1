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
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.num_epochs):
            for x_i, y_i in zip(X, y):
                y_pred = np.sign(np.dot(self.w, x_i) + self.b)

                if y_pred != y_i:
                    self.w = self.w + y_i * x_i
                    self.b = self.b + y_i

    def predict(self, X):
        """
        TODO: Compute w*x + b. Return +1 or -1.
        """
        scores = X @ self.w + self.b
        return np.sign(scores)