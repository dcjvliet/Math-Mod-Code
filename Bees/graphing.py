import numpy as np
import math
from matplotlib import pyplot as plt


def egg_laying_rate(t: float):
    anchor_days = [0, 15, 74, 105, 166, 197, 228, 300, 350]
    anchor_rates = [41.67, 83.3, 0, 400, 1800, 1600, 1300, 0, 0]
    return np.interp(t % 365, anchor_days, anchor_rates, period=365)


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

    raise ValueError(f"Invalid day of year: {t_new}")


def find_weibull_survival(age: float, mean: float, std: float):
    exponent = (std / mean) ** -1.086
    lambda_param = mean / math.gamma(1 + (1 / exponent))
    return np.exp(-(age / lambda_param) ** exponent)


def find_population(t: int, gender: str = 'female'):
    populations_history = []
    if gender == 'female':
        for current_day in range(t):
            population = 0
            for birth_day in range(current_day + 1):
                population += egg_laying_rate(birth_day) * find_weibull_survival(current_day - birth_day, get_female_means(birth_day), get_female_stds(birth_day))
            populations_history.append(population)
    else:
        for current_day in range(t):
            population = 0
            for birth_day in range(current_day + 1):
                population += egg_laying_rate(birth_day) * find_weibull_survival(current_day - birth_day, 26.5, 5.5)
            populations_history.append(population)

    return populations_history


def initial_population(t: int, starting_population: int):
    populations_history = []
    for current_day in range(t):
        population = starting_population * find_weibull_survival(current_day, 150, 30)
        populations_history.append(population)
    return populations_history


def get_female_means(t: int):
    anchor_means = [150, 150, 150, 22.37, 19.29, 23.84, 22.84, 13.58, 26.25, 150, 150, 150]
    anchor_points = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    return np.interp(t % 365, anchor_points, anchor_means, period=365)


def get_female_stds(t: int):
    anchor_stds = [30, 30, 30, 1.89, 4.64, 3.91, 3.6, 2.06, 5.08, 30, 30, 30]
    anchor_points = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    return np.interp(t % 365, anchor_points, anchor_stds, period=365)


if __name__ == '__main__':
    TIME_SPAN = 730 # days
    INITIAL_POPULATION = 6000
    FEMALE_PROPORTION = 0.925
    MALE_PROPORTION = 0.075

    # find female population and scale by FEMALE_PROPORTION to account for male:female ratio
    female_population_history = find_population(TIME_SPAN, gender='female')
    female_population_history = [x * FEMALE_PROPORTION for x in female_population_history]

    # find male population and scale by MALE_PROPORTION to account for male:female ratio
    male_population_history = find_population(TIME_SPAN, gender='male')
    male_population_history = [x * MALE_PROPORTION for x in male_population_history]

    # find initial population and how it falls off over time
    initial_population_history = initial_population(TIME_SPAN, INITIAL_POPULATION)

    # find total population of the hive
    total_population_history = [female + male + initial for female, male, initial in zip(female_population_history, male_population_history, initial_population_history)]
    
    # all initial bees are female, so add to the female population history (after calculating total so we don't double count)
    female_population_history = [female + initial for female, initial in zip(female_population_history, initial_population_history)]

    # graph
    plt.plot(female_population_history, label='Female')
    plt.plot(male_population_history, label='Male')
    plt.plot(total_population_history, label='Total')
    plt.title('Bee Population Over Time')
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.legend()
    plt.xlim(365, TIME_SPAN)
    plt.show()
    