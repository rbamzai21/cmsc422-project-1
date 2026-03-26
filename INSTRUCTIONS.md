# Project 1: Fundamentals of Classification [100 pts total]

The goal of this project is to implement from scratch the main Machine Learning techniques we have learned so far. You will build these classifiers, evaluate them, and conduct an analysis. Along the way, you will answer specific questions about *how* the data looks, *why* models make mistakes, and *how* hyperparameters change the performance of the models.

## Introduction and Setup

You can get the necessary files by heading to the Files page, selecting the p1 folder and clicking the Download as Zip button. Then you can unzip it on your machine and work inside that directory. 

### What and how to Submit

You will submit the assignment on Gradescope.
* **Upload the python files you will edit**, namely `simple_classifier.py`, `knn.py`, `perceptron.py`, `dt.py`, as well as `main.py` and `utils.py` (so all python files we gave you except `tests.py`, which you will not need to change). Please note that even if the code passes the tests from `tests.py` this is *not* a guarantee for its corectness.
* **Include a PDF file named `REPORT.pdf`** that answers all the written questions in the assignment marked by "Q#" below.

Note that the file `main.py` is a scaffold used to generate your numbers and plots. You may work in a group of **at most 2 students**, or you can work individually if you so choose. 

**Only one group member should submit the assignment as a zipped folder** on Gradescope, but make sure to add the names of all group members in the dedicated tab on Gradescope when submitting.

### Setup Instructions
You should start by installing Conda. You can find the Conda installation guide online. After installing Conda, you should see "(base)" in your terminal.

To create a Conda environment for Project 1, run the following command in the terminal:
`conda create -n cmsc422_proj1 python=3.11.7`

To activate the environment, run:
`conda activate cmsc422_proj1`

Now, install the required libraries by running:
`pip install numpy==2.0.1 matplotlib==3.9.2 scikit-learn==1.6.1 pandas==3.0.1`


## Part 1: A Simple Classifier and Data Processing [20 pts]

Let's begin our foray into classification by preparing our data and looking at a very simple classifier. Before coding, look at what you are dealing with. Open `main.py`, where we have provided a call to fetch the famous MNIST dataset of handwritten digits using `scikit-learn`. The images are 28x28 pixels, flattened into a vector of 784 numbers. 

Before we can train models on this data, we need to process it. We want to filter the dataset to only include the digits "3" and "8" to create a binary classification problem. We will map the label "8" to **+1** and the label "3" to **-1**.

In `utils.py`, complete the `process_data(X, y)` function [3 pts]. You will need to:
1. Filter the arrays to keep only the 3s and 8s.
2. Map the string labels '8' to `1.0` and '3' to `-1.0`.
3. Normalize the pixel values in `X` to be between `0.0` and `1.0` (they start as 0-255).
4. Use `sklearn`'s `train_test_split` to split the data into 80% training and 20% testing (use `random_state=42` for reproducibility).

You will also need to implement the `compute_accuracy(y_true, y_pred)` function in `utils.py` [3 pts], which will be used to evaluate all models throughout this project. Test your implementation by running `python tests.py`. 

Once your tests pass, implement the TODOs in `analysis_part1` from `main.py` and run the code to answer Q1, Q2, and Q3 below. 

* **Q1 (Report): [2 pts]** Plot a 3 and an 8 from the dataset using the `plot_images` function. Include the image in your report.
* **Q2 (Report): [4 pts]** How many training and testing examples for the task of classifying between "3" vs "8" are there? How many features (dimensions) does each example have?
* **Q3 (Report): [4 pts]** Count the number of positive labels (+1 for "8") and negative labels (-1 for "3") in the training and testing sets.

Your next implementation task will be to fill in the missing functionality in `MostFrequentClassClassifier` inside `simple_classifier.py`. It should remember whether +1 or -1 is more common in training and always predict that. Test it by running `python tests.py`. 

* **Q4 (Report): [4 pts]** Run your `MostFrequentClassClassifier` on the 3 vs 8 Data (see the relevant TODO in `main.py`). What is the test accuracy? Is this a good classifier in general? Why or why not?


## Part 2: K-Nearest Neighbors [30 pts]

For this section, we will switch from MNIST to the **CIFAR-10** dataset. CIFAR-10 consists of 32x32 color images across 10 classes. However, we will tackle a binary classification problem using only two of the ten classes: **'airplane'** and **'frog'**. Each example is flattened into a vector of 3072 features (32 x 32 x 3 color channels).

You will implement and use the K-Nearest Neighbors Classifier for this part.

First, implement `compute_euclidean_distance` [4 pts] and then implement `KNN.get_k_neighbors_indices` [5 pts] and `KNN.predict` [5 pts] in `knn.py`. Test your logic by running `python tests.py`. 

Once your code is working, use `main.py` to fetch and process the CIFAR-10 data. We have provided `fetch_cifar10`, `process_cifar_data`, and `plot_image_and_neighbors` in `utils.py`. The processing function will filter the dataset to only include airplanes and frogs, and map the labels to `-1.0` and `+1.0` respectively. *Note:* Because standard KNN can be very slow to compute on large datasets, `process_cifar_data` will automatically shuffle and subset the data to just 10% of the filtered images. This ensures your code runs relatively fast.

Train a KNN with `k=5` on the **Airplane vs Frog** data in `main.py`. Find a test example where your model makes a **Correct** prediction. Use your model's `get_k_neighbors_indices` method and the `plot_image_and_neighbors` function to show the test image alongside its 5 nearest neighbors in the training set.

* **Q5 (Report): [4 pts]** Include the plot of the correct test sample and its 5 neighbors. Analyze the plot. Do the neighbors look visually similar to the test image? What features (e.g., color, shapes, background) do you think the distance metric is picking up on?

Next, find a test example where the model makes a **Mistake**. Plot this Test Image and its 5 nearest neighbors. 

* **Q6 (Report): [4 pts]** Include the plot of the misclassified test sample and its 5 neighbors. Why do you think the model got confused? What do the neighbors have in common with the test image that might have mislead the Euclidean distance?

Finally, evaluate the effect of the hyperparameter K. Run KNN on the CIFAR-10 data for `k = [3, 5, 7, 9, 11, 13]`. Compute the training and testing accuracies for each value. Plot these accuracies on a single graph (X-axis = K, Y-axis = Accuracy).

* **Q7 (Report): [4 pts]** Include your accuracy plot. Discuss the trends: where do you observe overfitting or underfitting as K changes? Explain why this happens based on the behavior of the KNN algorithm.
* **Q8 (Report): [4 pts]** What would happen if you set K equal to the total number of training samples ($K = N_{train}$)? What is the conceptual connection between a KNN model with $K = N_{train}$ and the `MostFrequentClassClassifier` from Part 1?


## Part 3: The Perceptron [25 pts]

In `perceptron.py`, you will implement the classic Perceptron Update Rule.

You’ll start by implementing `Perceptron.predict` [5 pts], which should return either **+1** or **-1** based on the linear combination of weights and inputs. Once that's set, move on to the `train` function. You must **initialize your weights and bias to zeros**. You will then loop through your data for the specified number of epochs, updating the weights whenever the model makes a mistake [8 pts].

After you've implemented the code, it is time to evaluate how the model behaves on different data distributions. We will start with a synthetic blob dataset provided in `utils.get_blob_data()`, which creates two distinct clusters of points.

* **Q9 (Report): [3 pts]** Train your Perceptron on the blob dataset for 50 epochs. What are your final training and testing accuracies? 

* **Q10 (Report): [3 pts]** Use `utils.plot_decision_boundary` to visualize your model's performance on the blob dataset. Include this plot in your report. 

Finally, let's explore a fundamental limitation of the Perceptron using a custom dataset of three collinear blobs via `utils.get_collinear_blobs()`. This consists of three clusters in a straight line: a middle cluster of one class between two clusters of another class.

* **Q11 (Report): [3 pts]** Train your Perceptron on this collinear dataset for 100 epochs. Include the image of the final decision boundary. What is the final training accuracy?

* **Q12 (Report): [3 pts]** Look closely at the data points in the plot. Explain conceptually why the Perceptron fails on this specific data.

## Part 4: Decision Trees [25 pts]

This will be your simple implementation of a decision tree classifier in `dt.py`. Your implementation should be based on the minimum classification error heuristic.

Because decision trees are highly interpretable, we will move away from the pixel data for this section. Instead, we will use datasets with explainable features provided in `datasets.py` (such as `TennisData` or `SentimentData`). 

Your task is to recursively build the tree based on the provided hints. Decision trees are stored as simple data structures. Each node has an `is_leaf` boolean that is true for leaves and false otherwise. Leaves have an assigned class label (+1 or -1). Internal nodes have a `feature` to split on, a `left` child for when the feature value is < 0.5, and a `right` child for when the feature value is >= 0.5.

Once you've implemented the training function `train_dt` [7 pts], you should implement `predict` [5 pts]. Traverse the tree based on the threshold of 0.5 for each feature. Train a tree with `max_depth: 1` on `datasets.TennisData`. This should produce a "decision stump" with one branch and two leaves. 

You can also print the tree and visualize what features have been used to split in each node. For example:
```python
h = dt.DT({"max_depth": 4})
h.train(datasets.TennisData.X, datasets.TennisData.Y)
print(h)
```
Will print:
```
Branch 6
  Branch 7
    Leaf 1
    Branch 2
      Leaf 1
      Leaf -1
  Branch 1
    Branch 7
      Branch 2
        Leaf -1
        Leaf 1
      Leaf -1
    Leaf 1
```

* **Q13 (Report) [3 pts]:** Next, evaluate your model's performance on `datasets.SentimentData` using trees with `max_depth` of 1, 3, and 5. Print the training and test accuracies for each of these depths in your analysis code. Make sure your printed output matches the following format:
`Training accuracy [value], test accuracy [value]`

For the next two questions, generate learning curves and hyperparameter curves by changing the dataset size and the `max_depth` hyperparameter in `main.py`.

* **Q14 (Report): [5 pts]** Plot the training and test accuracies as a function of **dataset size** (train only the first `N` training examples for `N` ranging from small numbers up to the dataset size, e.g. `[1, 5, 10, 20, 50, 100, 200, 500, datasets.SentimentData.X.shape[0]]`). Why does training accuracy tend to go *down*? What happens to the testing accuracy and why? Why is the test curve often "jagged" toward the left? 
* **Q15 (Report): [5 pts]** Plot the training and test accuracies as a fuctnion of the **tree's depth**, e.g. use varying depths such as `[1, 3, 5, 7, 11, 15, 20, 30]`. Varying `max_depth` on the Sentiment Data will cause training accuracy increases monotonically, but test accuracy usually forms a "hill". Which of these is *guaranteed* to happen and which is just expected? Why? 
