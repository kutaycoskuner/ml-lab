"""
Two-dimensional synthetic data generators.

This module provides functions for generating simple datasets
that can be used to test and demonstrate machine learning models.
"""

import numpy as np


def generate_linear_data(
    start=1,
    stop=10,
    num_points=50,
    slope=5,
    intercept=30,
    noise=3,
    seed=None,
):
    """
    Generate synthetic data following a linear relationship.

    The underlying relationship is:

        y = slope * x + intercept

    Random noise is added to make the data more realistic.

    Returns
    -------
    x : numpy.ndarray
        One-dimensional input values.

    y : numpy.ndarray
        One-dimensional target values.
    """

    if seed is not None:
        np.random.seed(seed)

    # Generate evenly spaced x-values
    x = np.linspace(start, stop, num_points)

    # Generate random noise
    random_noise = np.random.normal(
        loc=0,
        scale=noise,
        size=num_points,
    )

    # Generate y-values
    y = slope * x + intercept + random_noise

    return x, y