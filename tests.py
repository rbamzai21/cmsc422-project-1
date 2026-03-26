import numpy as np
import unittest

import datasets
import utils
import simple_classifier
import knn
import perceptron
import dt


class TestProject1(unittest.TestCase):

    # ==========================================
    # Part 1: Simple Classifiers & Data Processing
    # ==========================================

    def test_part1_process_data(self):
        print("\n[Part 1] Testing process_data...")
        
        # 1. Deterministic but mixed up mock data.
        # Exactly 10 valid examples ('3' and '8') and 4 invalid ones ('0', '1', '5', '9')
        y_mock = np.array(['3', '8', '5', '3', '8', '1', '3', '8', '3', '8', '9', '3', '8', '0'])
        
        # 2. X_mock with min (0) and max (255) values explicitly included to test normalization
        X_mock = np.array([
            [255, 0], [0, 255], [10, 10], [128, 128], [50, 50], [0, 0], 
            [255, 255], [100, 200], [0, 10], [20, 30], [99, 99], [255, 128], 
            [128, 255], [5, 5]
        ])
        
        result = utils.process_data(X_mock, y_mock)
        
        if result[0] is None:
            self.fail("UNIMPLEMENTED: utils.process_data() is currently returning None.")
            
        X_tr, X_te, y_tr, y_te = result
        
        # General Property: Did we keep exactly the right number of total valid examples? (10)
        self.assertEqual(len(y_tr) + len(y_te), 10, 
                         "Failed: Did not correctly filter out all invalid digits ('0', '1', '5', '9').")
        
        # General Property: Are ALL features strictly normalized between 0.0 and 1.0?
        self.assertTrue(np.all(X_tr >= 0.0) and np.all(X_tr <= 1.0), 
                        "Failed: Not all X_train features are properly normalized in the [0.0, 1.0] range.")
        self.assertTrue(np.all(X_te >= 0.0) and np.all(X_te <= 1.0), 
                        "Failed: Not all X_test features are properly normalized in the [0.0, 1.0] range.")
        
        # General Property: Are the labels mapped to strictly +1.0 and -1.0?
        unique_labels = set(np.unique(np.concatenate((y_tr, y_te))))
        self.assertTrue(unique_labels.issubset({1.0, -1.0}), 
                        f"Failed: Labels contain invalid values. Expected subset of {{1.0, -1.0}}, but found: {unique_labels}")
        
        # General Property: Is the 80/20 split mathematically correct? (80% of 10 = 8 train, 2 test)
        self.assertEqual(len(y_tr), 8, "Failed: Incorrect train split size (should be 8).")
        self.assertEqual(len(y_te), 2, "Failed: Incorrect test split size (should be 2).")
        print("PASS")
        
    def test_part1_accuracy(self):
        print("\n[Part 1] Testing compute_accuracy...")
        
        # Hardcoded Example 1 (Small)
        y_t1 = np.array([1, -1, 1, 1, -1])
        y_p1 = np.array([1, -1, -1, 1, 1])
        acc1 = utils.compute_accuracy(y_t1, y_p1)
        
        if acc1 == 0.0:
            self.fail("UNIMPLEMENTED: utils.compute_accuracy() is returning the default 0.0.")
            
        self.assertEqual(acc1, 0.60, "Failed: Accuracy calculation is incorrect for small arrays.")
        
        # Hardcoded Example 2 (Longer)
        y_t2 = np.array([ 1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1])
        y_p2 = np.array([ 1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1])
        
        acc2 = utils.compute_accuracy(y_t2, y_p2)
        self.assertAlmostEqual(acc2, 0.70, places=5, 
                               msg="Failed: Accuracy calculation is incorrect for longer arrays.")
        print("PASS")

    def test_part1_most_frequent(self):
        print("\n[Part 1] Testing MostFrequentClassClassifier...")
        np.random.seed(42)
        X = np.random.rand(15, 2)
        
        # Create labels: 11 of class -1, 4 of class 1.
        y = np.array([-1.0] * 11 + [1.0] * 4)
        
        # Shuffle so they aren't grouped together linearly
        np.random.shuffle(y) 
        
        clf = simple_classifier.MostFrequentClassClassifier()
        clf.train(X, y)
        
        if clf.prediction == 0:
            self.fail("UNIMPLEMENTED: MostFrequentClassClassifier.train() did not update self.prediction.")
            
        preds = clf.predict(np.random.rand(5, 2)) # Test on 5 mock examples
        
        if np.all(preds == 0):
            self.fail("UNIMPLEMENTED: MostFrequentClassClassifier.predict() is returning default zeros.")
            
        self.assertEqual(clf.prediction, -1.0, "Failed: Classifier did not select the true majority class (-1.0).")
        self.assertEqual(len(preds), 5, "Failed: Predict should return an array matching the length of the input X.")
        self.assertTrue(np.all(preds == -1.0), "Failed: Predict should return the majority class for ALL examples.")
        print("PASS")


    # ==========================================
    # Part 2: K-Nearest Neighbors
    # ==========================================

    def test_part2_knn_distance(self):
        print("\n[Part 2] Testing KNN Euclidean Distance...")
        x1 = np.array([1, 1])
        x2 = np.array([4, 5])
        dist = knn.compute_euclidean_distance(x1, x2)
        
        if dist == 0.0:
            self.fail("UNIMPLEMENTED: knn.compute_euclidean_distance() is returning the default 0.0.")
            
        self.assertEqual(dist, 5.0, "Failed: Euclidean distance calculation is incorrect.")
        print("PASS")

    def test_part2_knn_get_indices(self):
        print("\n[Part 2] Testing KNN get_k_neighbors_indices...")
        X_train = np.array([[0, 0], [1, 1], [10, 10], [11, 11], [2, 2]])
        y_train = np.array([-1.0, 1.0, -1.0, 1.0, -1.0])
        
        model = knn.KNN(k=3)
        model.train(X_train, y_train)
        
        x_single = np.array([0.9, 0.9])
        
        try:
            indices = model.get_k_neighbors_indices(x_single)
        except Exception as e:
            self.fail(f"get_k_neighbors_indices threw an exception: {e}")
            
        if len(indices) == 0:
            self.fail("UNIMPLEMENTED: knn.KNN.get_k_neighbors_indices() is returning an empty list/array.")
            
        self.assertEqual(len(indices), 3, "Failed: get_k_neighbors_indices did not return exactly k indices.")
        
        expected_indices = {0, 1, 4}
        student_indices = set(indices)
        self.assertEqual(student_indices, expected_indices, f"Failed: get_k_neighbors_indices returned incorrect indices. Expected {expected_indices}, got {student_indices}.")
        print("PASS")

    def test_part2_knn_predict_binary(self):
        print("\n[Part 2] Testing KNN Binary Predict (Majority Vote)...")
        # Dataset with distinct negative and positive clusters
        X_train = np.array([
            [0, 0], [0, 0.1], [0, -0.1],   # Cluster near origin (-1.0)
            [5, 5], [5, 5.1], [5, 4.9]     # Cluster near 5,5 (+1.0)
        ])
        
        y_train = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        
        model = knn.KNN(k=3)
        model.train(X_train, y_train)
        
        if model.X_train is None:
            self.fail("UNIMPLEMENTED: knn.KNN.train() did not store the training data.")
        
        X_test = np.array([[0.05, 0], [5.05, 5.0]])
        
        try:
            preds = model.predict(X_test)
        except Exception as e:
            self.fail(f"predict threw an exception: {e}")
        
        if len(preds) == 0:
            self.fail("UNIMPLEMENTED: knn.KNN.predict() is returning an empty array.")
            
        # First point should be -1.0, second should be +1.0
        expected_preds = np.array([-1.0, 1.0])
        self.assertTrue(np.array_equal(preds, expected_preds), 
                        f"Failed: KNN predictions are incorrect. Expected {expected_preds}, got {preds}")
        print("PASS")


    # ==========================================
    # Part 3: Perceptron
    # ==========================================

    def test_part3_perceptron_weights_and_bias(self):
        print("\n[Part 3] Testing Perceptron Update Rule (Weights & Bias)...")
        # We supply one single training point to guarantee one update step
        X = np.array([[1.0, 2.0]])
        y = np.array([1.0])
        
        p = perceptron.Perceptron(num_epochs=1)
        p.train(X, y)
        
        if p.w is None:
            self.fail("UNIMPLEMENTED: perceptron.Perceptron.train() did not initialize weights (self.w).")
            
        # Initial prediction = w*x + b = 0*1 + 0*2 + 0 = 0
        # Mistake! Since y*pred <= 0  (1.0 * 0 <= 0)
        # Update rule: w = w + y*x  => w = [0,0] + 1.0*[1.0, 2.0] = [1.0, 2.0]
        #              b = b + y    => b = 0 + 1.0 = 1.0
        
        expected_w = np.array([1.0, 2.0])
        expected_b = 1.0
        
        self.assertTrue(np.array_equal(p.w, expected_w), 
                        f"Failed: Perceptron Weights not updated correctly. Expected {expected_w}, got {p.w}")
        
        self.assertEqual(p.b, expected_b, 
                         f"Failed: Perceptron Bias not updated correctly. Expected {expected_b}, got {p.b}")
        print("PASS")

    def test_part3_perceptron_train_and_predict(self):
        print("\n[Part 3] Testing Perceptron on Linearly Separable Data (AND Gate)...")
        X = np.array([
            [0, 0], 
            [0, 1], 
            [1, 0], 
            [1, 1]
        ])
        y = np.array([-1.0, -1.0, -1.0, 1.0]) 
        
        p = perceptron.Perceptron(num_epochs=10)
        p.train(X, y)
            
        preds = p.predict(X)
        if np.all(preds == 0):
            self.fail("UNIMPLEMENTED: perceptron.Perceptron.predict() is returning default zeros.")
            
        acc = utils.compute_accuracy(y, preds)
        self.assertEqual(acc, 1.0, f"Failed: Perceptron did not correctly separate the data. Expected accuracy 1.0, got {acc}")
        print("PASS")

    # ==========================================
    # Part 4: Decision Trees
    # ==========================================

    def test_part4_dt_train_predict(self):
        print("\n[Part 4] Testing Decision Trees accuracy on simple data...")
        # Use a real explainable dataset
        X = datasets.TennisData.X
        y = datasets.TennisData.Y
        
        # Initialize DT with max_depth of 1 (Decision Stump)
        model = dt.DT({"max_depth": 1}) 
        model.train(X, y)
        
        preds = model.predict(X)
        acc = utils.compute_accuracy(y, preds)
        
        # A simple stump on TennisData should do better than random chance
        # We ensure it runs, splits on a feature, and returns valid predictions
        self.assertGreater(acc, 0.6, "Decision Tree stump failed to learn a meaningful split on TennisData.")
        self.assertTrue(np.all(np.isin(preds, [-1, 1])), "Predictions must be exactly +1 or -1.")
        print("PASS")

if __name__ == '__main__':
    unittest.main(verbosity=0)