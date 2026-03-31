"""
boundary_experiments.py

Explores how the covariance matrices of a 2-class Gaussian generative model
affect the shape of the decision boundary in 2D.

When both classes share the same covariance, the boundary is always linear
(it's the perpendicular bisector of the line connecting the two means, roughly).
But when the covariances differ, you can get all sorts of conic section boundaries:
circles, ellipses, parabolas, hyperbolas.

This is because the decision boundary is where:
    log pi_0 + log P_0(x) = log pi_1 + log P_1(x)

Expanding the Gaussian log-pdfs, you get a quadratic equation in x, and the
solutions to a quadratic in 2D are exactly the conic sections.

We keep the means fixed at (5, 0) and (-5, 0) with equal priors, and just
vary the covariance matrices to get different boundary shapes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def generate_data(pi, mu, sigma, n):
    """
    Sample n points from a Gaussian mixture model.
    Each point gets a class label drawn from pi, then its
    coordinates are drawn from the corresponding Gaussian.
    """
    k = len(pi)
    x = np.zeros((n, 2))
    y = np.random.choice(k, n, p=pi)

    for i in range(n):
        c = y[i]
        x[i, :] = np.random.multivariate_normal(mu[c], sigma[c])

    return x, y


def predict_on_grid(pi, mu, sigma, grid):
    """
    For every point on the grid, compute log pi_j + log P_j(x) for each class
    and return the argmax. This tells us which class "owns" each grid cell.
    """
    k = len(pi)
    gn = grid.shape[0]
    scores = np.zeros((gn, k))

    for j in range(k):
        rv = multivariate_normal(mean=mu[j], cov=sigma[j])
        for i in range(gn):
            scores[i, j] = np.log(pi[j]) + rv.logpdf(grid[i, :])

    return np.argmax(scores, axis=1)


def show_boundary(pi, mu, sigma, x, y, title="", colors=('red', 'blue')):
    """
    Visualize the decision boundary by coloring a fine grid according to
    which class the model assigns, then overlay the actual data points.
    """
    delta = 0.05

    # figure out a reasonable bounding box from the data
    x0min, x0max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x1min, x1max = x[:, 1].min() - 1, x[:, 1].max() + 1

    # create the grid
    xx, yy = np.meshgrid(
        np.arange(x0min, x0max, delta),
        np.arange(x1min, x1max, delta)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # classify every grid point
    Z = predict_on_grid(pi, mu, sigma, grid)
    Z = Z.reshape(xx.shape)

    # plot the colored regions
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.tab20c, vmin=-2, vmax=2, shading='auto')

    # scatter the data points on top
    for i in range(len(y)):
        plt.plot(x[i, 0], x[i, 1], 'o', color=colors[y[i]], markersize=2)

    plt.xlim(x0min, x0max)
    plt.ylim(x1min, x1max)
    plt.axis('equal')
    if title:
        plt.title(title, fontsize=12)


def run_experiment(sigma, title, filename):
    """
    Run a single experiment: generate data, plot the boundary, save the figure.
    Uses fixed means at (5,0) and (-5,0) with equal priors.
    """
    pi = np.array([0.5, 0.5])
    mu = np.array([[5.0, 0.0], [-5.0, 0.0]])

    x, y = generate_data(pi, mu, sigma, 1000)

    plt.figure(figsize=(8, 6))
    show_boundary(pi, mu, sigma, x, y, title=title)
    plt.tight_layout()
    plt.savefig(f"outputs/{filename}", dpi=150)
    plt.show()
    print(f"  Saved to outputs/{filename}")


if __name__ == "__main__":
    np.random.seed(42)

    # ---------------------------------------------------------------
    # Experiment 1: Linear boundary (identity covariance for both)
    # When Sigma_0 = Sigma_1, the quadratic terms cancel and we
    # get a linear boundary. With identity covariance, it's just
    # the y-axis (perpendicular bisector of the two means).
    # ---------------------------------------------------------------
    print("1. Linear boundary (axis-parallel, identity covariances)")
    sigma_identity = np.array([
        [[1, 0], [0, 1]],
        [[1, 0], [0, 1]]
    ])
    run_experiment(sigma_identity, "Linear boundary (identity covariance)", "boundary_linear_identity.png")

    # ---------------------------------------------------------------
    # Experiment 2: Linear boundary, but tilted
    # Still Sigma_0 = Sigma_1, so the boundary stays linear. But now
    # the shared covariance has off-diagonal terms, which rotates
    # the boundary away from the coordinate axes.
    # ---------------------------------------------------------------
    print("\n2. Linear boundary (tilted, shared non-diagonal covariance)")
    sigma_tilted = np.array([
        [[2, 0.5], [0.5, 1]],
        [[2, 0.5], [0.5, 1]]
    ])
    run_experiment(sigma_tilted, "Linear boundary (tilted)", "boundary_linear_tilted.png")

    # ---------------------------------------------------------------
    # Experiment 3: Spherical (circular) boundary
    # If the covariances are different multiples of the identity,
    # the boundary becomes a circle. The class with bigger variance
    # "spreads out" more, so the tighter class wins near its mean
    # and the spread-out class wins everywhere else.
    # ---------------------------------------------------------------
    print("\n3. Spherical (circular) boundary")
    sigma_spherical = np.array([
        [[16, 0], [0, 16]],
        [[1, 0], [0, 1]]
    ])
    run_experiment(sigma_spherical, "Spherical boundary", "boundary_spherical.png")

    # ---------------------------------------------------------------
    # Experiment 4: Elliptical boundary
    # Same idea as spherical, but the larger covariance isn't
    # isotropic — it's stretched more in one direction. This turns
    # the circle into an ellipse.
    # ---------------------------------------------------------------
    print("\n4. Elliptical boundary")
    sigma_elliptical = np.array([
        [[16, 0], [0, 2]],
        [[1, 0], [0, 1]]
    ])
    run_experiment(sigma_elliptical, "Elliptical boundary", "boundary_elliptical.png")

    # ---------------------------------------------------------------
    # Experiment 5: Hyperbolic boundary
    # This is the trickiest one. We need the two covariances to be
    # "stretched" in different directions. Class 0 is tall and narrow,
    # class 1 is wide and short. The boundary curves away in two
    # directions, forming a hyperbola.
    # ---------------------------------------------------------------
    print("\n5. Hyperbolic boundary")
    sigma_hyperbolic = np.array([
        [[1, 0], [0, 4]],
        [[16, 0], [0, 1]]
    ])
    run_experiment(sigma_hyperbolic, "Hyperbolic boundary", "boundary_hyperbolic.png")

    print("\nAll done! Check the outputs/ folder for the plots.")
