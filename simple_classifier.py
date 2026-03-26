import numpy as np

class MostFrequentClassClassifier:
    def __init__(self):
        self.prediction = 0

    def train(self, X, y):
        """
        TODO: Find the most frequent label in y and store it in self.prediction.
        """
        values, counts = np.unique(y, return_counts=True)
        self.prediction = values[np.argmax(counts)]

    def predict(self, X):
        """
        TODO: Return a vector of predictions, all equal to self.prediction.
        """
        return np.full(X.shape[0], self.prediction)