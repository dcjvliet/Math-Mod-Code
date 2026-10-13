import numpy as np
from scipy import stats


def calc_regression_line(aa_x, aa_y, ax_x, ax_y):
    aa_slope, aa_intercept, _, _, _ = stats.linregress(aa_x, aa_y)
    ax_slope, ax_intercept, _, _, _ = stats.linregress(ax_x, ax_y)
    slope = (aa_slope + ax_slope) / 2 # type: ignore
    intercept = (aa_intercept + ax_intercept) / 2 # type: ignore
    return slope, intercept


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

aa_points = []
ax_points = []
for fly in aa:
    # do wing length (x) and antenna length (y)
    aa_points.append((fly[0], fly[2]))

for fly in ax:
    # do wing length (x) and antenna length (y)
    ax_points.append((fly[0], fly[2]))

# now we want to find a linear regression line for each
aa_x_values = [point[0] for point in aa_points]
aa_y_values = [point[1] for point in aa_points]
ax_x_values = [point[0] for point in ax_points]
ax_y_values = [point[1] for point in ax_points]

slope, intercept = calc_regression_line(aa_x_values, aa_y_values, ax_x_values, ax_y_values)

# now we want to update with each new fly
rng = np.random.default_rng()
aa_measurements = aa[:, [0, 2]]
ax_measurements = ax[:, [0, 2]]

aa_mean = aa_measurements.mean(axis=0)
ax_mean = ax_measurements.mean(axis=0)

aa_cov = np.cov(aa_measurements, rowvar=False)
ax_cov = np.cov(ax_measurements, rowvar=False)

correct_count = 0
incorrect_count = 0
for _ in range(2500):
    aa_wing_length, aa_antenna_length = rng.multivariate_normal(
        aa_mean, aa_cov
    )
    ax_wing_length, ax_antenna_length = rng.multivariate_normal(
        ax_mean, ax_cov
    )

    # now we see if it falls above (aa) or below (ax) the regression line
    expected_value = slope * aa_wing_length + intercept
    expected_value_ax = slope * ax_wing_length + intercept
    if aa_antenna_length > expected_value:
        correct_count += 1
    else:
        incorrect_count += 1

    if ax_antenna_length < expected_value_ax:
        correct_count += 1
    else:
        incorrect_count += 1

    # now we want to update the regression
    aa_x_values.append(aa_wing_length)
    aa_y_values.append(aa_antenna_length)
    ax_x_values.append(ax_wing_length)
    ax_y_values.append(ax_antenna_length)
    slope, intercept = calc_regression_line(aa_x_values, aa_y_values, ax_x_values, ax_y_values)

# print out accuracy
print(f"Correct: {correct_count}, Incorrect: {incorrect_count}")
print(f"Accuracy: {correct_count / (correct_count + incorrect_count) * 100:.2f}%")