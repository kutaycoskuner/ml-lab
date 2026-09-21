"""
linear_regression.py

A reusable implementation of simple linear regression.

INPUT
-----
The model requires two one-dimensional numeric datasets:

    x : input / independent variable
        Example: study hours, house size, temperature

    y : target / dependent variable
        Example: exam score, house price, energy consumption

x and y must:
    - contain numeric values
    - have the same number of values
    - contain at least two data points

Each x[i] corresponds to y[i].

The model assumes a linear relationship:

    y_hat = w * x + b


OUTPUT
------
fit(x, y) returns:

    w   : optimal slope
    b   : optimal intercept
    mse : Mean Squared Error

plot_regression() saves a PNG file containing:

    - the original generated data points
    - the fitted regression line
"""

import matplotlib.pyplot as plt
import numpy as np


def fit(x, y):
    """
    Calculate the analytical least-squares solution.

    Parameters
    ----------
    x : array-like
        One-dimensional numeric input data.

    y : array-like
        One-dimensional numeric target data.

    Returns
    -------
    w : float
        Optimal slope.

    b : float
        Optimal intercept.

    mse : float
        Mean Squared Error.
    """

    # Convert the generated data to NumPy arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Check that x and y are one-dimensional
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional.")

    # Check that x and y have the same number of values
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    # At least two data points are required
    if len(x) < 2:
        raise ValueError("At least two data points are required.")

    # Calculate the means
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate deviations from the means
    x_deviation = x - x_mean
    y_deviation = y - y_mean

    # Calculate the numerator and denominator
    # of the analytical solution for w
    numerator = np.sum(x_deviation * y_deviation)
    denominator = np.sum(x_deviation ** 2)

    # All x values cannot be identical
    if denominator == 0:
        raise ValueError("x must contain more than one unique value.")

    # Calculate the optimal slope
    w = numerator / denominator

    # Calculate the optimal intercept
    b = y_mean - w * x_mean

    # Calculate predictions for the generated data
    y_pred = w * x + b

    # Calculate Mean Squared Error
    mse = np.mean((y - y_pred) ** 2)

    return w, b, mse


def plot_regression(x, y, w, b, output_file="linear_regression.png"):
    """
    Plot the generated data and fitted regression line.

    The graph is saved as a PNG file.
    """

    # Convert the generated data to NumPy arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Create smooth x-values for the regression line
    x_line = np.linspace(x.min(), x.max(), 100)

    # Calculate y-values using the fitted model
    y_line = w * x_line + b

    # Plot the generated data
    plt.scatter(
        x,
        y,
        label="Generated data",
    )

    # Plot the regression line
    plt.plot(
        x_line,
        y_line,
        label=f"Regression: y = {w:.2f}x + {b:.2f}",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression")
    plt.legend()

    # Save the graph
    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    # Close the figure
    plt.close()