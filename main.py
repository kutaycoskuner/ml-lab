import os

from src.data_generators.two_dimensional import generate_linear_data
from src.models.linear_regression import fit, plot_regression


def main():
    # Generate synthetic data
    x, y = generate_linear_data(
        start=1,
        stop=10,
        num_points=50,
        slope=5,
        intercept=30,
        noise=3,
        seed=42,
    )

    # Fit the linear regression model
    w, b, mse = fit(x, y)

    # Print the results
    print("Slope (w):", w)
    print("Intercept (b):", b)
    print("MSE:", mse)

    # Create the output directory if it does not exist
    output_directory = "output"
    os.makedirs(output_directory, exist_ok=True)

    # Save the regression plot
    output_file = os.path.join(
        output_directory,
        "linear_regression.png",
    )

    plot_regression(x, y, w, b, output_file)

    print("Plot saved to:", output_file)


if __name__ == "__main__":
    main()