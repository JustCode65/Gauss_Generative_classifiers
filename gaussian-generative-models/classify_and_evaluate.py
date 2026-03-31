"""
classify_and_evaluate.py

Takes the trained Gaussian generative model and evaluates it on the MNIST test set.
Reports the overall error rate and shows some randomly chosen misclassified examples
to get a feel for what kinds of mistakes the model makes.

Run fit_and_regularize.py first to generate model_params.pkl.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from mnist_data_loader import load_test_set
from fit_and_regularize import load_model


def predict_test_set(test_data, pi, mu, sigma, k=10):
    """
    Run the generative model on every test image.
    Returns the predicted labels.
    """
    n = test_data.shape[0]
    scores = np.zeros((n, k))

    for j in range(k):
        rv = multivariate_normal(mean=mu[j], cov=sigma[j])
        # logpdf can handle the whole batch at once, which is nice
        scores[:, j] = np.log(pi[j]) + rv.logpdf(test_data)

    return np.argmax(scores, axis=1)


def show_misclassified_samples(test_data, test_labels, predictions, num_samples=5):
    """
    Randomly pick some misclassified digits and display them.
    It's interesting to see what trips up the model — often the
    mistakes are on digits that are genuinely ambiguous or weird.
    """
    wrong_indices = np.where(predictions != test_labels)[0]

    # pick a random subset
    np.random.seed(0)
    chosen = np.random.choice(wrong_indices, size=min(num_samples, len(wrong_indices)), replace=False)

    fig, axes = plt.subplots(1, len(chosen), figsize=(3 * len(chosen), 3))
    if len(chosen) == 1:
        axes = [axes]

    for i, idx in enumerate(chosen):
        ax = axes[i]
        ax.imshow(test_data[idx].reshape(28, 28), cmap=plt.cm.gray)
        ax.set_title(f"true: {test_labels[idx]}, pred: {predictions[idx]}")
        ax.axis('off')

    plt.suptitle("Misclassified examples", fontsize=14)
    plt.tight_layout()
    plt.savefig("outputs/misclassified_samples.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # load the saved model
    try:
        pi, mu, sigma, c = load_model()
    except FileNotFoundError:
        print("No saved model found. Run fit_and_regularize.py first!")
        exit(1)

    print(f"Loaded model with c = {c}")

    # load test set
    test_data, test_labels = load_test_set()
    print(f"Test set: {len(test_labels)} images\n")

    # classify everything
    print("Classifying test images...")
    predictions = predict_test_set(test_data, pi, mu, sigma)

    # tally up the results
    errors = np.sum(predictions != test_labels)
    error_rate = errors / len(test_labels)
    print(f"\nResults: {errors} errors out of {len(test_labels)}")
    print(f"Error rate: {error_rate:.4f} ({error_rate * 100:.2f}%)")

    # break it down by digit — which ones does the model struggle with?
    print("\nPer-digit error rates:")
    for j in range(10):
        mask = (test_labels == j)
        digit_errors = np.sum(predictions[mask] != j)
        digit_total = np.sum(mask)
        print(f"  digit {j}: {digit_errors:3d} / {digit_total:4d} wrong  ({digit_errors/digit_total*100:.1f}%)")

    # show some misclassified examples
    print("\nShowing 5 random misclassified digits:")
    show_misclassified_samples(test_data, test_labels, predictions, num_samples=5)
    print("Saved to outputs/misclassified_samples.png")
