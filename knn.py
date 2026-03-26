import numpy as np

def compute_euclidean_distance(x1, x2):
    """
    TODO: Compute Euclidean distance between x1 and x2.
    """
    return 0.0

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        """
        TODO: Store the training data.
        """
        pass

    def get_k_neighbors_indices(self, x_single):
        """
        TODO: 
        Given a single test example x_single, calculate its distance to every point 
        in self.X_train and return the INDICES of the k nearest neighbors.
        """
        return []

    def predict(self, X):
        """
        TODO:
        1. Initialize empty predictions.
        2. Loop through every input example in X.
        3. For each example:
           a. Use get_k_neighbors_indices to find the k nearest neighbors.
           b. Get the labels of those neighbors.
           c. Vote (Majority wins). 
        """
        predictions = []
        # Your code here...
        return np.array(predictions)