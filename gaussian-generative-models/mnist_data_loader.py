"""
mnist_data_loader.py

Handles loading the MNIST dataset from the raw gzipped IDX files.
Also has a quick helper to display individual digits.
"""

import gzip
import numpy as np
import matplotlib.pyplot as plt
import os


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_images(filename):
    """
    Reads a gzipped IDX file of images and returns them as a (N, 784) array.
    Each row is a flattened 28x28 grayscale image, pixel values 0-255.
    """
    filepath = os.path.join(DATA_DIR, filename)
    with gzip.open(filepath, 'rb') as f:
        # first 16 bytes are magic number, num images, rows, cols — skip them
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 784)
    return data


def load_labels(filename):
    """
    Reads a gzipped IDX file of labels and returns them as a 1D array.
    Each entry is an integer 0-9.
    """
    filepath = os.path.join(DATA_DIR, filename)
    with gzip.open(filepath, 'rb') as f:
        # first 8 bytes are magic number and count — skip them
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


def load_training_set():
    """Load the full 60k training images and labels."""
    images = load_images("train-images-idx3-ubyte.gz")
    labels = load_labels("train-labels-idx1-ubyte.gz")
    return images, labels


def load_test_set():
    """Load the 10k test images and labels."""
    images = load_images("t10k-images-idx3-ubyte.gz")
    labels = load_labels("t10k-labels-idx1-ubyte.gz")
    return images, labels


def display_digit(image, title=None):
    """
    Show a single MNIST digit. Pass in a 784-dim vector
    and this reshapes it back to 28x28 for display.
    """
    plt.imshow(np.reshape(image, (28, 28)), cmap=plt.cm.gray)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()


if __name__ == "__main__":
    # just a quick sanity check — load the data and show a random digit
    train_data, train_labels = load_training_set()
    test_data, test_labels = load_test_set()

    print(f"Training set: {train_data.shape[0]} images, each {train_data.shape[1]} pixels")
    print(f"Test set:     {test_data.shape[0]} images, each {test_data.shape[1]} pixels")

    # show a random training example
    idx = np.random.randint(0, len(train_labels))
    print(f"\nShowing training image #{idx}, label = {train_labels[idx]}")
    display_digit(train_data[idx])
