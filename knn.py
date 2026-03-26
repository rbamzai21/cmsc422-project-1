import numpy as np

def compute_euclidean_distance(x1, x2):
    """
    TODO: Compute Euclidean distance between x1 and x2.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2)) 

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        """
        TODO: Store the training data.
        """
        self.X_train = X
        self.y_train = y

    def get_k_neighbors_indices(self, x_single):
        """
        TODO: 
        Given a single test example x_single, calculate its distance to every point 
        in self.X_train and return the INDICES of the k nearest neighbors.
        """
        distances = np.array([
            compute_euclidean_distance(x_single, x_train)
            for x_train in self.X_train
        ])
        return np.argsort(distances)[:self.k]

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
        for x_single in X:
            neighbor_indices = self.get_k_neighbors_indices(x_single)
            neighbor_labels = self.y_train[neighbor_indices]
            vote = np.sum(neighbor_labels)
            predictions.append(1.0 if vote > 0 else -1.0)
        return np.array(predictions)