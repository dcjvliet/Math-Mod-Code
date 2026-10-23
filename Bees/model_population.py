import math
import calendar

import numpy as np
import matplotlib.pyplot as plt


def display_graph(female_population: list[float], male_population: list[float], timespan: int, last_year_only: bool = False, show_month_ticks: bool = True, save_path: str = '') -> None:
    """A function for graphing the bee population over time.

    :param female_population: The female bee population over time
    :type female_population: list[float]
    :param male_population: The male bee population over time
    :type male_population: list[float]
    :param timespan: The number of days to graph data for
    :type timespan: int
    :param last_year_only: If true, it only graphs the last year of data, defaults to False
    :type last_year_only: bool, optional
    :param show_month_ticks: If true, it shows ticks on the x-axis for each month instead of the number of days, defaults to True
    :type show_month_ticks: bool, optional
    :param save_path: The path to save the graph (if not specified, the graph is not saved), defaults to ''
    :type save_path: str, optional
    """
    # logic for getting x-axis date values
    days = range(max(len(female_population), len(male_population))) # these should be the same, but jic

    plt.plot(days, female_population, label='Female Population', color='red', linewidth=2.5)
    plt.plot(days, male_population, label='Male Population', color='blue', linewidth=2.5)
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('Bee Population Over Time')
    plt.legend()
    if last_year_only:
        plt.xlim(days[max(0, timespan - 365)], timespan) # in case there isn't a year of data

    # show ticks for each month
    if show_month_ticks:
        month_names = list(calendar.month_abbr)[1:]
        month_starts = []
        month_labels = []
        month_start_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

        for day in range(timespan + 1):
            year_day = day % 365

            if year_day in month_start_days:
                month_starts.append(day)
                month_labels.append(month_names[month_start_days.index(year_day)])

        plt.xticks(month_starts, month_labels, rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def get_egg_laying_rate(t: float) -> float:
    """Get the egg laying rate at a given time

    :param t: The time since january 1st in days
    :type t: float
    :return: The egg laying rate (per day)
    :rtype: float
    """
    anchor_days = [0, 15, 74, 105, 166, 197, 228, 300, 350]
    anchor_rates = [41.67, 83.3, 0, 400, 1800, 1600, 1300, 0, 0]
    return np.interp(t % 365, anchor_days, anchor_rates, period=365)


def get_female_means(t: float) -> float:
    """Get the mean lifespan for a female bee at the specified time

    :param t: The time since January 1st in days
    :type t: float
    :return: The mean lifespan for a female bee at the specified time
    :rtype: float
    """
    anchor_means = [150, 150, 150, 22.37, 19.29, 23.84, 22.84, 13.58, 26.25, 150, 150, 150]
    anchor_points = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    return np.interp(t % 365, anchor_points, anchor_means, period=365)


def get_female_stds(t: float) -> float:
    """Get the standard deviation in lifespan for a female bee at the specified time

    :param t: The time since January 1st in days
    :type t: float
    :return: The standard deviation in lifespan for a female bee at the specified time
    :rtype: float
    """
    anchor_stds = [30, 30, 30, 1.89, 4.64, 3.91, 3.6, 2.06, 5.08, 30, 30, 30]
    anchor_points = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    return np.interp(t % 365, anchor_points, anchor_stds, period=365)


def day_to_month(t: float) -> int:
    """Convert a day to a month (1-12)

    :param t: The time since january 1st in days
    :type t: float
    :raises ValueError: If the day is not valid (should never happen)
    :return: The month (1-12)
    :rtype: int
    """
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


def find_weibull_survival_probability(age: float, mean: float, std: float) -> float:
    """A function to find the probability of survival at a given time using the Weibull survival function

    :param age: The age of the bee in days
    :type age: float
    :param mean: The average lifespan of the bee in days
    :type mean: float
    :param std: The standard deviation of the lifespan of the bee in days
    :type std: float
    :return: The probability of survival at the given age
    :rtype: float
    """
    exponent = (std / mean) ** -1.086
    lambda_param = mean / math.gamma(1 + (1 / exponent))
    return np.exp(-(age / lambda_param) ** exponent)


def find_population(t: int, gender: str = 'female') -> list[float]:
    """Find the population of a given bee gender at a given time

    :param t: The time since January 1st (in days)
    :type t: int
    :param gender: The gender of the bees to consider, defaults to 'female'
    :type gender: str, optional, 'female' or 'male'
    :return: The population of bees at each time step
    :rtype: list[float]
    """
    populations_history = []
    if gender == 'female':
        for current_day in range(t):
            population = 0
            for birth_day in range(current_day + 1):
                population += get_egg_laying_rate(birth_day) * find_weibull_survival_probability(current_day - birth_day, get_female_means(birth_day), get_female_stds(birth_day))
            populations_history.append(population)
    else:
        for current_day in range(t):
            population = 0
            for birth_day in range(current_day + 1):
                # male bees we use the same mean of 26.5 and standard deviation of 5.5 for the entire year
                population += get_egg_laying_rate(birth_day) * find_weibull_survival_probability(current_day - birth_day, 26.5, 5.5)
            populations_history.append(population)

    return populations_history


def initial_population(t: int, starting_population: float) -> list[float]:
    """Find the population of the initial bees at a given time

    :param t: The number of days since January 1st (in days)
    :type t: int
    :param starting_population: The initial population of bees
    :type starting_population: float
    :return: A list of population values for each day
    :rtype: list[float]
    """
    populations_history = []
    for current_day in range(t):
        # find probability for each day, no new initial bees are ever born
        population = starting_population * find_weibull_survival_probability(current_day, 150, 30)
        populations_history.append(population)
    return populations_history


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
    display_graph(female_population_history, male_population_history, TIME_SPAN, last_year_only=False, show_month_ticks=True)
    