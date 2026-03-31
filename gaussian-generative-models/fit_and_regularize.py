"""
fit_and_regularize.py

Fits a Gaussian generative model to the MNIST training data.

The big idea: model each digit class j as a multivariate Gaussian N(mu_j, Sigma_j),
with prior probability pi_j. To classify a new image x, pick the digit j that
maximizes log(pi_j) + log P_j(x).

The catch is that the raw 784x784 covariance matrices are basically always singular
(way more dimensions than data points per class), so we regularize by adding c*I
to each covariance. The constant c is chosen by validation.

This script:
  1. Splits the 60k training set into 50k train / 10k validation
  2. Computes per-class means and covariances from the 50k
  3. Sweeps over a range of c values, evaluating each on the validation set
  4. Picks the c with lowest validation error
  5. Saves the fitted model parameters to disk

Speed trick: we eigendecompose each covariance matrix once. Then for any
regularization constant c, adding cI just shifts the eigenvalues by c,
so we can evaluate the logpdf very cheaply without re-decomposing.
"""

import numpy as np
from scipy.stats import multivariate_normal
import pickle
import time

from mnist_data_loader import load_training_set, display_digit


def compute_class_parameters(x, y, k=10):
    """
    Given training data x (N x 784) and labels y (N,), compute the
    ML estimates of the class priors, means, and covariances.

    Returns:
        pi:    (k,) array of class frequencies
        mu:    (k, 784) array of class means
        sigma: (k, 784, 784) array of class covariance matrices (unregularized)
    """
    n, d = x.shape
    mu = np.zeros((k, d))
    sigma = np.zeros((k, d, d))
    pi = np.zeros(k)

    for j in range(k):
        # grab all images with label j
        mask = (y == j)
        x_j = x[mask]

        pi[j] = x_j.shape[0] / n  # fraction of training set that's digit j
        mu[j] = np.mean(x_j, axis=0)

        # center the data before computing covariance
        centered = x_j - mu[j]
        sigma[j] = np.dot(centered.T, centered) / x_j.shape[0]

    return pi, mu, sigma


def classify(x, pi, mu, sigma, k=10):
    """
    Classify a batch of images using the generative model.
    For each image, pick the label j that maximizes log(pi_j) + log P_j(x).

    x:     (N, 784) test images
    pi:    (k,) class priors
    mu:    (k, 784) class means
    sigma: (k, 784, 784) class covariances (already regularized)

    Returns predicted labels as a (N,) array.
    """
    n = x.shape[0]
    scores = np.zeros((n, k))

    for j in range(k):
        # multivariate_normal handles the log-pdf computation for us,
        # which is crucial — the raw probabilities in 784 dimensions
        # would underflow to zero instantly
        rv = multivariate_normal(mean=mu[j], cov=sigma[j])
        scores[:, j] = np.log(pi[j]) + rv.logpdf(x)

    return np.argmax(scores, axis=1)


def precompute_eigen(mu, sigma_raw, x_val, k=10):
    """
    Eigendecompose each covariance matrix and project the validation data
    into each class's eigenbasis. This lets us evaluate the logpdf for
    any regularization constant c without re-decomposing.

    The Gaussian logpdf is:
        -0.5 * [d*log(2pi) + sum(log(eig_i + c)) + sum(z_i^2 / (eig_i + c))]

    where z = Q^T (x - mu), Q is the eigenvector matrix, and eig_i are eigenvalues.

    We precompute z^2 for each class, so the c-sweep is just fast arithmetic.
    """
    eigenvalues = []   # eigenvalues for each class
    z_squared = []     # projected & squared deviations for each class

    print("  Eigendecomposing covariance matrices...")
    for j in range(k):
        # symmetric matrix, so use eigh (faster and more stable than eig)
        eigvals, eigvecs = np.linalg.eigh(sigma_raw[j])

        # project validation data into this eigenbasis
        # z[i, :] = Q^T (x_val[i] - mu[j])
        centered = x_val - mu[j]           # (N, 784)
        z = centered @ eigvecs              # (N, 784)
        zsq = z ** 2                        # (N, 784)

        eigenvalues.append(eigvals)
        z_squared.append(zsq)

    return eigenvalues, z_squared


def fast_classify_for_c(pi, eigenvalues, z_squared, c, k=10):
    """
    Classify validation points for a specific c value using precomputed
    eigendecompositions. Way faster than rebuilding the scipy distribution.
    """
    d = len(eigenvalues[0])
    n = z_squared[0].shape[0]
    scores = np.zeros((n, k))

    log_2pi = np.log(2 * np.pi)

    for j in range(k):
        shifted_eigs = eigenvalues[j] + c  # (784,)

        # log determinant = sum of log eigenvalues
        log_det = np.sum(np.log(shifted_eigs))

        # mahalanobis distance = sum(z_i^2 / eigenvalue_i) for each point
        mahal = np.sum(z_squared[j] / shifted_eigs, axis=1)  # (N,)

        # full logpdf (the -d/2 * log(2pi) part doesn't affect the argmax
        # but including it for completeness)
        logpdf = -0.5 * (d * log_2pi + log_det + mahal)

        scores[:, j] = np.log(pi[j]) + logpdf

    return np.argmax(scores, axis=1)


def search_for_best_c(x_train, y_train, x_val, y_val):
    """
    Try a range of regularization constants and return the one that
    gives the lowest validation error.

    Uses eigendecomposition to make the sweep fast — decompose each
    covariance once, then evaluating for a new c is just arithmetic.
    """
    k = 10
    print("Computing class parameters from training set...")
    pi, mu, sigma_raw = compute_class_parameters(x_train, y_train)

    # precompute the eigendecompositions and projections
    eigenvalues, z_squared = precompute_eigen(mu, sigma_raw, x_val, k)

    # sweep over c values: {1, 51, 101, ..., 10001}
    # this is feasible now because each evaluation is fast
    c_values = list(range(1, 10002, 50))
    errors = []

    print(f"Sweeping over {len(c_values)} values of c...")

    best_c = None
    best_err = 1.0

    for i, c in enumerate(c_values):
        preds = fast_classify_for_c(pi, eigenvalues, z_squared, c, k)
        err = np.sum(preds != y_val) / len(y_val)
        errors.append(err)

        if err < best_err:
            best_err = err
            best_c = c

        if (i + 1) % 40 == 0 or i == 0:
            print(f"  c = {c:6d}  |  val error = {err:.4f}  |  best so far: c = {best_c}, err = {best_err:.4f}")

    print(f"\nDone! Best c = {best_c} with validation error = {best_err:.4f}")
    return best_c, c_values, errors, pi, mu, sigma_raw


def save_model(pi, mu, sigma, c, filepath="model_params.pkl"):
    """Save the fitted model to disk so we don't have to retrain."""
    model = {
        'pi': pi,
        'mu': mu,
        'sigma': sigma,  # the regularized version
        'c': c,
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath="model_params.pkl"):
    """Load a previously saved model."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model['pi'], model['mu'], model['sigma'], model['c']


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # load the full 60k training set
    train_data, train_labels = load_training_set()
    print(f"Loaded {len(train_labels)} training images.\n")

    # split into 50k training / 10k validation
    # shuffle first so the split is random
    np.random.seed(42)  # for reproducibility
    perm = np.random.permutation(len(train_labels))
    train_data = train_data[perm]
    train_labels = train_labels[perm]

    x_train = train_data[:50000]
    y_train = train_labels[:50000]
    x_val = train_data[50000:]
    y_val = train_labels[50000:]

    print(f"Training split: {len(y_train)} images")
    print(f"Validation split: {len(y_val)} images\n")

    # search for the best regularization constant
    t0 = time.time()
    best_c, c_values, val_errors, pi, mu, sigma_raw = search_for_best_c(
        x_train, y_train, x_val, y_val
    )
    elapsed = time.time() - t0
    print(f"Search took {elapsed:.1f} seconds.\n")

    # plot the validation error curve
    plt.figure(figsize=(8, 5))
    plt.plot(c_values, val_errors, linewidth=1.5)
    plt.xlabel("c")
    plt.ylabel("Validation error")
    plt.title("Regularization constant vs. validation error")
    plt.tight_layout()
    plt.savefig("outputs/validation_curve.png", dpi=150)
    plt.show()
    print("Saved validation curve to outputs/validation_curve.png")

    # save the final model with the best c
    d = mu.shape[1]
    sigma_final = sigma_raw + best_c * np.eye(d)
    save_model(pi, mu, sigma_final, best_c)

    # show what the learned means look like — they should resemble
    # "average" versions of each digit
    print("\nClass means (these should look like blurry digits):")
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for j in range(10):
        ax = axes[j // 5, j % 5]
        ax.imshow(mu[j].reshape(28, 28), cmap=plt.cm.gray)
        ax.set_title(f"digit {j}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("outputs/class_means.png", dpi=150)
    plt.show()
    print("Saved class means to outputs/class_means.png")
