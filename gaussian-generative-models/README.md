# Gaussian Generative Models

Exploring Gaussian generative models for classification — from simple 2D toy examples
to full handwritten digit recognition on MNIST.

## What's in here

- **`mnist_data_loader.py`** — Helper functions to load and inspect the MNIST dataset
- **`fit_and_regularize.py`** — Fitting Gaussian generative models to MNIST with covariance regularization, plus a validation-based search for the best regularization constant `c`
- **`classify_and_evaluate.py`** — Running the trained model on the test set, computing error rates, and visualizing misclassified digits
- **`boundary_experiments.py`** — Playing with 2D Gaussian generative models to see how different covariance matrices produce different decision boundary shapes (linear, spherical, elliptical, hyperbolic)
- **`study_guide.md`** — A walkthrough of the key ideas and takeaways

## Setup

```bash
pip install numpy scipy matplotlib
```

Put the four MNIST `.gz` files in the `data/` directory (they should already be there).

## Quick start

```bash
# Run the full MNIST pipeline (takes a few minutes due to validation search)
python fit_and_regularize.py
python classify_and_evaluate.py

# Run the 2D boundary experiments (fast, produces plots)
python boundary_experiments.py
```

## Results

The Gaussian generative model achieves roughly **4.25% test error** on MNIST with
a regularization constant around `c = 2151`. 
