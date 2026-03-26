import os
import urllib.request
import tarfile
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs


def process_data(X, y):
    """
    Processes the raw MNIST dataset into a clean binary classification task.
    
    Args:
        X (numpy.ndarray): The raw image data. Shape is (70000, 784). 
                           Each row represents a single image, flattened 
                           from a 28x28 grid into a 784-element array. 
                           Values range from 0 to 255.
        y (numpy.ndarray): The raw labels. Shape is (70000,). 
                           These are stored as string characters from '0' to '9'.
                           
    TODO:
    1. Filter X and y: Create a boolean mask to keep only the examples 
       where the label in y is '3' or '8'.
    2. Map the labels: Convert the string label '8' to the float 1.0, and 
       the string label '3' to the float -1.0. (Hint: np.where() is helpful).
    3. Normalize the features: Divide the pixel values of your filtered X 
       by 255.0 so that all feature values fall between 0.0 and 1.0.
    4. Split the data: Use sklearn's train_test_split to divide the 
       filtered data into 80% training and 20% testing sets. 
       (Make sure to use random_state=42 for grading reproducibility).
    
    Returns:
        X_train (numpy.ndarray): Training images, shape (N_train, 784).
        X_test (numpy.ndarray): Testing images, shape (N_test, 784).
        y_train (numpy.ndarray): Training labels (+1.0 or -1.0), shape (N_train,).
        y_test (numpy.ndarray): Testing labels (+1.0 or -1.0), shape (N_test,).
    """
    # Your code here...
    mask = (y == '3') | (y == '8')
    X_filtered = X[mask]
    y_filtered = y[mask]

    y_mapped = np.where(y_filtered == '8', 1.0, -1.0)

    X_normalized = X_filtered / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_mapped, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def compute_accuracy(y_true, y_pred):
    """
    Computes the accuracy of a model's predictions.
    
    Args:
        y_true (numpy.ndarray): The actual ground-truth labels. Shape is (N,).
        y_pred (numpy.ndarray): The labels predicted by the model. Shape is (N,).
        
    TODO: 
    Compare the predicted labels to the true labels. 
    Accuracy = (number of correct predictions) / (total number of predictions)
    (Hint: You can compare arrays directly using '==' and use np.sum() or np.mean())
    
    Returns:
        float: The accuracy value between 0.0 and 1.0.
    """
    # Your code here...
    return np.mean(y_true == y_pred)

def plot_images(img1, title1, img2, title2):
    """
    Plots two MNIST images side by side.
    
    Args:
        img1 (numpy.ndarray): A 784-dimensional vector representing the first image.
        title1 (str): Title for the first image.
        img2 (numpy.ndarray): A 784-dimensional vector representing the second image.
        title2 (str): Title for the second image.
    """
    fig, ax = plt.subplots(1, 2)
    # Reshaping the 784-dim vectors back to 28x28 grids for visualization
    ax[0].imshow(img1.reshape(28, 28), cmap='gray')
    ax[0].set_title(title1)
    ax[1].imshow(img2.reshape(28, 28), cmap='gray')
    ax[1].set_title(title2)
    plt.show()

def plot_image_and_neighbors(test_img, neighbor_imgs, title="Test Image and Neighbors"):
    """
    Plots a CIFAR-10 test image alongside its K nearest neighbors.
    
    Args:
        test_img (numpy.ndarray): A 3072-dimensional vector of the test image.
        neighbor_imgs (list/ndarray): A list of 3072-dimensional vectors of the neighbors.
        title (str): Title for the entire figure.
    """
    k = len(neighbor_imgs)
    fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 3))
    fig.suptitle(title, fontsize=16)
    
    # OpenML CIFAR-10 arrays are flattened channels (RRR...GGG...BBB...)
    # Reshape to (3, 32, 32) and transpose to (32, 32, 3) for matplotlib
    def format_img(img_vector):
        return img_vector.reshape(3, 32, 32).transpose(1, 2, 0)
    
    axes[0].imshow(format_img(test_img))
    axes[0].set_title("Test Image")
    axes[0].axis('off')
    
    for i in range(k):
        axes[i+1].imshow(format_img(neighbor_imgs[i]))
        axes[i+1].set_title(f"Neighbor {i+1}")
        axes[i+1].axis('off')
        
    plt.tight_layout()
    plt.show()

def fetch_cifar10():
    """
    Downloads the official CIFAR-10 dataset directly from U of Toronto.
    Returns the full dataset (60,000 images, 10 classes).
    """
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    extract_folder = "cifar-10-batches-py"

    if not os.path.exists(extract_folder):
        print(f"Downloading CIFAR-10 from {url} (this takes a minute)...")
        urllib.request.urlretrieve(url, filename)
        print("Extracting files...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()

    X_list, y_list = [], []
    for i in range(1, 6):
        with open(os.path.join(extract_folder, f"data_batch_{i}"), 'rb') as fo:
            batch_dict = pickle.load(fo, encoding='bytes')
            X_list.append(batch_dict[b'data'])
            y_list.append(batch_dict[b'labels'])

    with open(os.path.join(extract_folder, "test_batch"), 'rb') as fo:
        batch_dict = pickle.load(fo, encoding='bytes')
        X_list.append(batch_dict[b'data'])
        y_list.append(batch_dict[b'labels'])

    X = np.vstack(X_list)
    y_int = np.concatenate(y_list)

    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    y = np.array([label_names[i] for i in y_int])

    return X, y

def process_cifar_data(X, y, subset_fraction=0.10):
    """
    Processes the raw CIFAR-10 dataset into a binary task: Airplane vs Frog.
    Subsets the filtered data to speed up KNN processing.
    """
    # 1. Filter X and y to keep only 'airplane' and 'frog'
    mask = (y == 'airplane') | (y == 'frog')
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    # 2. Map labels: 'frog' -> +1.0, 'airplane' -> -1.0
    y_mapped = np.where(y_filtered == 'frog', 1.0, -1.0)
    
    # 3. Normalize to [0.0, 1.0]
    X_normalized = X_filtered / 255.0
    
    # 4. Shuffle and Subset (Ensuring reproducible random shuffle)
    np.random.seed(42) 
    shuffle_indices = np.random.permutation(len(X_normalized))
    X_shuffled = X_normalized[shuffle_indices]
    y_shuffled = y_mapped[shuffle_indices]
    
    subset_size = int(len(X_shuffled) * subset_fraction)
    X_subset = X_shuffled[:subset_size]
    y_subset = y_shuffled[:subset_size]
    
    # 5. Split 80/20 (This results in 960 train, 240 test for 10% fraction)
    return train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)




def get_blob_data(n_samples=2000):
    """
    Generates a dataset using two blobs.
    Maps labels to +1 and -1 and returns standard train/test split.
    """
    X, y = make_blobs(n_samples=n_samples, centers=2, random_state=42, cluster_std=3.0)
    y = np.where(y == 0, -1.0, 1.0)
    return train_test_split(X, y, test_size=0.5, random_state=42)

def get_collinear_blobs(n_per_blob=100, std=0.5, random_state=42):
    """
    Returns a dataset of three collinear blobs where the middle blob 
    belongs to a different class than the outer two.
    
    Returns
    -------
    X : np.ndarray of shape (3 * n_per_blob, 2)
    y : np.ndarray of shape (3 * n_per_blob,)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Three centers in a straight diagonal line
    centers = np.array([
        [0, 0],
        [2, 2],
        [4, 4]
    ])

    # Interleaved labels: -1, 1, -1
    labels = np.array([-1.0, 1.0, -1.0])

    X_list = []
    y_list = []

    for center, label in zip(centers, labels):
        blob = center + np.random.randn(n_per_blob, 2) * std
        X_list.append(blob)
        y_list.append(np.full(n_per_blob, label))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    return X, y

def plot_decision_boundary(X, y, model, title="Decision Boundary"):
    """
    Plots the data points and the decision boundary of a trained model.
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02
    
    # Generate grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.5)
    markers = ['o', '^']
    for i, cls in enumerate([-1.0, 1.0]):
        plt.scatter(
            X[y == cls, 0],
            X[y == cls, 1],
            c=y[y == cls],
            cmap=plt.cm.RdBu,
            vmin=y.min(),
            vmax=y.max(),
            marker=markers[i],
            edgecolors='k',
            label=f'Class {cls}'
        )
    plt.title(title)
    plt.show()