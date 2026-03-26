import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

import datasets
import utils
import simple_classifier
import knn
import perceptron
import dt

def analysis_part1():
    print("Part 1: Loading and Processing Data...")
    # Fetch raw data
    print("Fetching MNIST data (this might take a few seconds)...")
    X_raw, y_raw = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    
    # Process the data using your implemented function in utils
    X_train, X_test, y_train, y_test = utils.process_data(X_raw, y_raw)
    
    # Q1: Plot a 3 and an 8
    # TODO: Find one example of a +1 (an '8') and one example of a -1 (a '3') in X_train.
    # Use utils.plot_images() to display them side-by-side.
    idx_3 = np.where(y_train == -1.0)[0][0]
    idx_8 = np.where(y_train == 1.0)[0][0]
    utils.plot_images(X_train[idx_3], "3", X_train[idx_8], "8")
    
    # Q2: Print the shapes of the training and testing sets.
    # TODO: Print the shape of X_train, y_train, X_test, and y_test.
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Q3: Print the number of positive (+1) and negative (-1) examples in both sets.
    # TODO: Count and print how many +1s and -1s are in y_train and y_test.
    print(f"Train -- +1 (8s): {np.sum(y_train == 1.0)}, -1 (3s): {np.sum(y_train == -1.0)}")
    print(f"Test  -- +1 (8s): {np.sum(y_test  == 1.0)},  -1 (3s): {np.sum(y_test  == -1.0)}")
    
    # Q4: Most Frequent Baseline Evaluation
    clf = simple_classifier.MostFrequentClassClassifier()
    clf.train(X_train, y_train)
    preds = clf.predict(X_test)
    
    # TODO: Compute and print the test accuracy of the MostFrequentClassClassifier.
    # Hint: Use utils.compute_accuracy(y_test, preds)
    acc = utils.compute_accuracy(y_test, preds)
    print(f"MostFrequentClassClassifier test accuracy: {acc:.4f}")

def analysis_part2():
    print("\nPart 2 (KNN on 10% of CIFAR-10):")
    
    # Fetch CIFAR-10 data
    print("Fetching CIFAR-10 dataset...")
    X_raw_cifar, y_raw_cifar = utils.fetch_cifar10()
    
    # Process the data
    X_train_c, X_test_c, y_train_c, y_test_c = utils.process_cifar_data(X_raw_cifar, y_raw_cifar)
    
    # Q5: Visualizing Neighbors for a Correct Prediction
    # TODO: Train KNN(k=5). 
    # Find a test example where the prediction is CORRECT.
    # Get its 5 nearest neighbors from the training set.
    # Plot using utils.plot_image_and_neighbors.
    
    model = knn.KNN(k=5)
    model.train(X_train_c, y_train_c)
    test_preds = model.predict(X_test_c)

    correct_indices = np.where(test_preds == y_test_c)[0]
    correct_idx = correct_indices[0]
    neighbor_indices = model.get_k_neighbors_indices(X_test_c[correct_idx])
    neighbor_imgs = X_train_c[neighbor_indices]
    true_label = "frog" if y_test_c[correct_idx] == 1.0 else "airplane"
    utils.plot_image_and_neighbors(
        X_test_c[correct_idx],
        neighbor_imgs,
        title=f"Q5: Correct Prediction — True Label: {true_label}"
    )

    # Q6: Visualizing Neighbors for a Mistake
    # TODO: Find a test example where the prediction is WRONG.
    # Get its 5 nearest neighbors from the training set.
    # Plot using utils.plot_image_and_neighbors.
    
    wrong_indices = np.where(test_preds != y_test_c)[0]
    wrong_idx = wrong_indices[0]
    neighbor_indices_wrong = model.get_k_neighbors_indices(X_test_c[wrong_idx])
    neighbor_imgs_wrong = X_train_c[neighbor_indices_wrong]
    true_label_wrong = "frog" if y_test_c[wrong_idx] == 1.0 else "airplane"
    pred_label_wrong = "frog" if test_preds[wrong_idx] == 1.0 else "airplane"
    utils.plot_image_and_neighbors(
        X_test_c[wrong_idx],
        neighbor_imgs_wrong,
        title=f"Q6: Wrong Prediction — True: {true_label_wrong}, Predicted: {pred_label_wrong}"
    )
 

    # Q7: Hyperparameters, Overfitting, and Underfitting
    k_vals = [3, 5, 7, 9, 11, 13]
    train_accs = ...
    test_accs = ...
    # TODO: Loop over k, train the model, get Train Acc and Test Acc.
    # Plot Train and Test accuracies vs. k.
    # Hint: plotting code below
     
    for k in k_vals:
        print(f"  Training KNN with k={k}...")
        m = knn.KNN(k=k)
        m.train(X_train_c, y_train_c)
        train_accs.append(utils.compute_accuracy(y_train_c, m.predict(X_train_c)))
        test_accs.append(utils.compute_accuracy(y_test_c, m.predict(X_test_c)))
        print(f"    k={k}  train={train_accs[-1]:.4f}  test={test_accs[-1]:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(k_vals, train_accs, marker='o', label='Train Accuracy')
    plt.plot(k_vals, test_accs, marker='s', label='Test Accuracy')
    plt.xlabel('K (Number of Neighbors)')
    plt.ylabel('Accuracy')
    plt.title('KNN Accuracy vs. K (Airplane vs Frog)')
    plt.legend()
    plt.grid(True)
    plt.show()

def analysis_part3():
    print("\nPart 3 (Perceptron):")
    
    # Q9 & Q10: Blob Dataset Analysis
    print("Fetching blob data...")
    X_train_blob, X_test_blob, y_train_blob, y_test_blob = utils.get_blob_data()
    
    # TODO: Train a Perceptron for 50 epochs on the blob training data.
    # TODO: Compute and print the final training and testing accuracies.
    # TODO: Use utils.plot_decision_boundary to visualize the model on the blob data.
    clf_blob = perceptron.Perceptron(num_epochs=50)
    clf_blob.train(X_train_blob, y_train_blob)

    train_acc = utils.compute_accuracy(y_train_blob, clf_blob.predict(X_train_blob))
    test_acc = utils.compute_accuracy(y_test_blob, clf_blob.predict(X_test_blob))
    print(f"Blob -- Train Accuracy: {train_acc:.4f}, Test: Accuracy: {test_acc:.4f}")

    utils.plot_decision_boundary(X_train_blob, y_train_blob, clf_blob, title="Perceptron Decision Boundary (Blobs)")
    
    # Q11: Collinear Blobs Problem
    print("\nFetching collinear data...")
    X_coll, y_coll = utils.get_collinear_blobs()

    # TODO: Train a Perceptron for 100 epochs on the collinear data.
    # TODO: Print the final training accuracy.
    # TODO: Use utils.plot_decision_boundary to visualize the model on the collinear data.
    clf_coll = perceptron.Perceptron(num_epochs=100)
    clf_coll.train(X_coll, y_coll)

    train_acc_coll = utils.compute_accuracy(y_coll, clf_coll.predict(X_coll))
    print(f"Collinear -- Train Accuracy: {train_acc_coll:.4f}")

    utils.plot_decision_boundary(X_coll, y_coll, clf_coll, title="Perceptron Decision Boundary (Collinear Blobs)")

def analysis_part4():
    print("\n--- Analysis Part 4 (Decision Trees) ---")
    
    # We use datasets.py for explainable features instead of MNIST/CIFAR
    tennis_X, tennis_y = datasets.TennisData.X, datasets.TennisData.Y          # train set
    tennis_Xte, tennis_yte = datasets.TennisData.Xte, datasets.TennisData.Yte  # test set

    sentiment_X, sentiment_y = datasets.SentimentData.X, datasets.SentimentData.Y
    sentiment_Xte, sentiment_yte = datasets.SentimentData.Xte, datasets.SentimentData.Yte
    
    # Q13: Evaluate performance with depths 1, 3, and 5 on SentimentData
    # TODO: Train DT with max_depth 1, 3, and 5 on sentiment_X/y. Evaluate and print accuracy.
    
    print("\nDecision Tree Depths from Sentiment Data:")
    for depth in [1, 3, 5]:
        model = dt.DT({"max_depth": depth})
        model.train(sentiment_X, sentiment_y)
        train_acc = utils.compute_accuracy(sentiment_y, model.predict(sentiment_X))
        test_acc  = utils.compute_accuracy(sentiment_yte, model.predict(sentiment_Xte))
        print(f"  max_depth={depth}: Training accuracy {train_acc:.4f}, test accuracy {test_acc:.4f}")
 

    # Q14: Learning Curves (Dataset Size)
    # TODO: Generate learning curves by changing the dataset size (e.g., using SentimentData).
    # Hint: use plotting code from above, you may also make it a function and call it from `utils`
    
    print("\nLearning Curves - Data Size ...")
    N_total = sentiment_X.shape[0]
    sizes = [1, 5, 10, 20, 50, 100, 200, 500, N_total]
    lc_train_accs = []
    lc_test_accs  = []
 
    for n in sizes:
        X_sub = sentiment_X[:n]
        y_sub = sentiment_y[:n]
        model = dt.DT({"max_depth": 5})
        model.train(X_sub, y_sub)
        lc_train_accs.append(utils.compute_accuracy(y_sub, model.predict(X_sub)))
        lc_test_accs.append( utils.compute_accuracy(sentiment_yte, model.predict(sentiment_Xte)))
 
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, lc_train_accs, marker='o', label='Train Accuracy')
    plt.plot(sizes, lc_test_accs,  marker='s', label='Test Accuracy')
    plt.xscale('log')
    plt.xlabel('Training Set Size (log scale)')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves — Decision Tree (max_depth=5) on SentimentData')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Q15: Hyperparameter Curves (Max Depth)
    # TODO: Generate hyperparameter curves by varying the max_depth hyperparameter on SentimentData.

    print("\nHyperparameter curves - Max Depth...")
    depths = [1, 3, 5, 7, 11, 15, 20, 30]
    hp_train_accs = []
    hp_test_accs  = []
 
    for depth in depths:
        model = dt.DT({"max_depth": depth})
        model.train(sentiment_X, sentiment_y)
        hp_train_accs.append(utils.compute_accuracy(sentiment_y,   model.predict(sentiment_X)))
        hp_test_accs.append( utils.compute_accuracy(sentiment_yte, model.predict(sentiment_Xte)))
        print(f"  max_depth={depth}: train={hp_train_accs[-1]:.4f}  test={hp_test_accs[-1]:.4f}")
 
    plt.figure(figsize=(8, 5))
    plt.plot(depths, hp_train_accs, marker='o', label='Train Accuracy')
    plt.plot(depths, hp_test_accs,  marker='s', label='Test Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Hyperparameter Curves — Decision Tree depth on SentimentData')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # You can comment/uncomment these out to run specific parts
    # analysis_part1()
    # analysis_part2()
    analysis_part3()
    # analysis_part4()
