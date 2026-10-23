import numpy as np
import math
from matplotlib import pyplot as plt


def egg_laying_rate(t: float):
    t_new = t % 365
    if 0 <= t_new < 15:
        return 41.67 + (t_new / 30) * 83.3
    if 15 <= t_new < 74:
        return 104.125 - 41.65 * (t_new / 30)
    if 74 <= t_new < 105:
        return 200 * (t_new / 30) - 500
    if 105 <= t_new < 166:
        return 800 * (t_new / 30) - 2600
    if 166 <= t_new < 197:
        return 300 * (t_new / 30) - 350
    if 197 <= t_new < 227:
        return 3550 - 300 * (t_new / 30)
    if 227 <= t_new < 330:
        return (-1300 / 3.5) * (t_new / 30) + (1300 / 3.5 * 11)
    if 330 <= t_new < 348:
        return 0
    else:
        return 83.3 * (t_new / 30) - 957.93


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


def initial_population(t: int, starting_population: int):
    populations_history = []
    for current_day in range(t):
        population = starting_population * find_proportion_alive(current_day, 0, [150] * 12, [30] * 12)
        populations_history.append(population)
    return populations_history

if __name__ == '__main__':
    female_means = [150, 150, 150, 22.37, 19.29, 23.84, 22.84, 13.58, 26.25, 150, 150, 150]
    female_stds = [30, 30, 30, 1.89, 4.64, 3.91, 3.6, 2.06, 5.08, 30, 30, 30]
    female_population_history = find_population(730, female_means, female_stds)
    female_population_history = [x * 0.925 for x in female_population_history]

    male_means = [26.5] * 12
    male_stds = [5.5] * 12
    male_population_history = find_population(730, male_means, male_stds)
    male_population_history = [x * 0.075 for x in male_population_history]

    initial_population_history = initial_population(730, 6000)

    total_population_history = [female + male + initial for female, male, initial in zip(female_population_history, male_population_history, initial_population_history)]
    # all initial are female so add afterwards so we don't double count
    female_population_history = [female + initial for female, initial in zip(female_population_history, initial_population_history)]

    plt.plot(female_population_history, label='Female')
    plt.plot(male_population_history, label='Male')
    plt.plot(total_population_history, label='Total')
    plt.title('Bee Population Over Time')
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.legend()
    plt.show()
    