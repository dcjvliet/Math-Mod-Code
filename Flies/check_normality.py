import matplotlib.pyplot as plt
import numpy as np
import scipy
from typing import cast


def check_normality(data: list[float]) -> None:
    if len(data) < 2:
        return

    (theoretical_quantiles, observed_values), (slope, intercept, _) = (
        scipy.stats.probplot(data, dist="norm")
    )
    theoretical_quantiles = np.asarray(theoretical_quantiles, dtype=float)
    observed_values = np.asarray(observed_values, dtype=float)
    slope = cast(float, slope)
    intercept = cast(float, intercept)
    fitted_values = slope * theoretical_quantiles + intercept

    plt.scatter(theoretical_quantiles, observed_values)
    plt.plot(
        theoretical_quantiles,
        fitted_values,
        color="red",
    )

    plt.xlabel("Z-Scores")
    plt.ylabel("Ax Abdomen Length")
    plt.title("Normal Q-Q Plot of Ax Abdomen Length")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    columns = ["wing", "abdomen", "antenna", "max_width"]

    aa = np.array([
        [2.10, 1.82, 1.39, 0.44],
        [3.31, 1.94, 1.58, 0.34],
        [3.83, 1.91, 1.64, 0.53],
        [3.11, 1.70, 1.45, 0.39],
        [1.65, 1.82, 1.38, 0.33],
        [1.73, 1.83, 1.54, 0.47],
        [1.84, 1.90, 1.45, 0.46],
        [2.49, 1.82, 1.61, 0.55],
        [2.53, 1.88, 1.48, 0.39],
        [2.99, 1.87, 1.62, 0.56]
    ])

    ax = np.array([
        [2.87, 1.78, 1.43, 0.46],
        [4.02, 1.86, 1.49, 0.38],
        [3.18, 1.96, 1.30, 0.34],
        [3.51, 2.00, 1.26, 0.47],
        [3.74, 2.05, 1.35, 0.43],
        [3.95, 1.96, 1.28, 0.33],
        [3.28, 1.87, 1.33, 0.46]
    ])

    # change this to get graphs for each atrtibute and type
    data = []
    for i in range(len(ax)):
        data.append(ax[i][1])

    check_normality(data)