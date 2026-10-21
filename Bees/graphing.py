import numpy as np
import math
from matplotlib import pyplot as plt


def egg_laying_rate(t: float):
    t_new = t % 365
    if 0 <= t_new < 31:
        return 83.3
    elif 31 <= t_new < 59:
        return 41.65
    elif 59 <= t_new < 90:
        return 0 
    elif 90 <= t_new < 120:
        return 200
    elif 120 <= t_new < 151:
        return 1000
    elif 151 <= t_new < 181:
        return 1300
    elif 181 <= t_new < 212:
        return 1600
    elif 212 <= t_new < 243:
        return 1300
    elif 243 <= t_new < 273:
        return 0
    elif 273 <= t_new < 304:
        return 0
    elif 304 <= t_new < 334:
        return 0
    elif 334 <= t_new < 365:
        return 0
    return -1


def time_to_month(t: float):
    t_new = t % 365
    if 0 <= t_new < 31:
        return 1
    elif 31 <= t_new < 59:
        return 2
    elif 59 <= t_new < 90:
        return 3
    elif 90 <= t_new < 120:
        return 4
    elif 120 <= t_new < 151:
        return 5
    elif 151 <= t_new < 181:
        return 6
    elif 181 <= t_new < 212:
        return 7
    elif 212 <= t_new < 243:
        return 8
    elif 243 <= t_new < 273:
        return 9
    elif 273 <= t_new < 304:
        return 10
    elif 304 <= t_new < 334:
        return 11
    elif 334 <= t_new < 365:
        return 12
    return -1


def _find_weibull_survival(age: float, mean: float, std: float):
    exponent = (std / mean) ** -1.086
    lambda_param = mean / math.gamma(1 + (1 / exponent))
    return np.exp(-(age / lambda_param) ** exponent)


def find_proportion_alive(age: float, birth_day: float, means: list[float], stds: list[float]):
    birth_month = time_to_month(birth_day)
    if birth_month == -1:
        return -1

    # Lifespan parameters describe the month in which the bee was born,
    # rather than the month represented by its current age.
    return _find_weibull_survival(age, means[birth_month - 1], stds[birth_month - 1])


def find_population(t: int, means: list[float], stds: list[float]):
    populations_history = []
    for current_day in range(t):
        population = 0
        for birth_day in range(current_day + 1):
            population += egg_laying_rate(birth_day) * find_proportion_alive(
                current_day - birth_day, birth_day, means, stds
            )
        populations_history.append(population)
    return populations_history


if __name__ == '__main__':
    means = [82.93, 62.74, 42.55, 22.37, 19.29, 23.84, 22.84, 13.58, 26.25, 143.5, 123.31, 103.12]
    stds = [13.58, 10.27, 6.97, 1.89, 4.64, 3.91, 3.6, 2.06, 5.08, 23.5, 20.19, 16.89]
    population_history = find_population(365, means, stds)

    plt.plot(population_history)
    plt.title('Bee Population Over Time')
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.show()
    