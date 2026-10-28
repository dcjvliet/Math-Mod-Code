import math
import calendar
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import PercentFormatter


class BeePopulationModel:
    def __init__(
                self,
                time_span: int = 730,
                initial_population: int = 6000,
                female_proportion: float = 0.925,
                male_proportion: float = 0.075,
                female_growth_duration: int = 21,
                male_growth_duration: int = 24,
                growth_success_rate: float = 0.85,
                egg_laying_multiplier: float = 1,
                zero_day: int = 300
            ):
        """Initialize a BeePopulationModel object

        :param time_span: The number of days to model the population for (starting on Jan 1), defaults to 730
        :type time_span: int, optional
        :param initial_population: The initial population of the bee colony, defaults to 6000
        :type initial_population: int, optional
        :param female_proportion: The proportion of females that will be born, defaults to 0.925
        :type female_proportion: float, optional
        :param male_proportion: The proportion of males that will be born, defaults to 0.075
        :type male_proportion: float, optional
        :param female_growth_duration: The number of days it takes for a female bee to mature, defaults to 21
        :type female_growth_duration: int, optional
        :param male_growth_duration: The number of days it takes for a male bee to mature, defaults to 24
        :type male_growth_duration: int, optional
        :param growth_success_rate: The chance a bee will survive to adulthood, defaults to 0.85
        :type growth_success_rate: float, optional
        :param egg_laying_multiplier: Multiplier for the egg laying rate (used for sensitivity analysis), defaults to 1
        :type egg_laying_multiplier: float, optional
        :param zero_day: The day when the egg laying rate hits zero in the winter (used for sensitivity analysis), defaults to 300
        :type zero_day: int, optional
        """
        self.TIME_SPAN = time_span # days
        self.INITIAL_POPULATION = initial_population
        self.FEMALE_PROPORTION = female_proportion
        self.MALE_PROPORTION = male_proportion
        self.FEMALE_GROWTH_DURATION = female_growth_duration
        self.MALE_GROWTH_DURATION = male_growth_duration
        self.GROWTH_SUCCESS_RATE = growth_success_rate
        self.EGG_LAYING_MULTIPLIER = egg_laying_multiplier
        self.ZERO_DAY = zero_day

    def display_graph(self, female_population: list[float], 
                    male_population: list[float], 
                    timespan: int, plot_total: bool = True, 
                    last_year_only: bool = False, 
                    show_month_ticks: bool = True, 
                    save_path: str = '', 
                    separate_by_life_stage: bool = False, 
                    baby_female_population: Optional[list[float]] = None, 
                    baby_male_population: Optional[list[float]] = None,
                    separate_graphs: bool = False) -> None:
        """A function for graphing the bee population over time.

        :param female_population: The female bee population over time (if separate_by_life_stage is True, this is the adult female population)
        :type female_population: list[float]
        :param male_population: The male bee population over time (if separate_by_life_stage is True, this is the adult male population)
        :type male_population: list[float]
        :param timespan: The number of days to graph data for
        :type timespan: int
        :param plot_total: If true, it plots the total population as well, defaults to True
        :type plot_total: bool, optional
        :param last_year_only: If true, it only graphs the last year of data, defaults to False
        :type last_year_only: bool, optional
        :param show_month_ticks: If true, it shows ticks on the x-axis for each month instead of the number of days, defaults to True
        :type show_month_ticks: bool, optional
        :param save_path: The path to save the graph (if not specified, the graph is not saved), defaults to ''
        :type save_path: str, optional
        :param separate_by_life_stage: If true, it separates the population by life stage, defaults to False
        :type separate_by_life_stage: bool, optional
        :param baby_female_population: The female baby bee population over time, defaults to None
        :type baby_female_population: list[float], optional
        :param baby_male_population: The male baby bee population over time, defaults to None
        :type baby_male_population: list[float], optional
        :param separate_graphs: If true, it creates separate graphs for each population, defaults to False
        :type separate_graphs: bool, optional
        :raises ValueError: If separate_by_life_stage is True, baby_female_population and baby_male_population must be provided
        """
        # logic for getting x-axis date values
        days = range(min(len(female_population), len(male_population))) # these should be the same, but jic

        if separate_graphs:
            if separate_by_life_stage and (baby_female_population is None or baby_male_population is None):
                raise ValueError("If separate_by_life_stage is True, baby_female_population and baby_male_population must be provided.")

            graph_data = [
                (female_population, baby_female_population, "Female", "red"),
                (male_population, baby_male_population, "Male", "blue"),
            ]

            if plot_total:
                total_adults = [female + male for female, male in zip(female_population, male_population)]

                total_babies = None
                if separate_by_life_stage:
                    total_babies = [female + male for female, male in zip(baby_female_population, baby_male_population)] # type: ignore

                graph_data.append(
                    (total_adults, total_babies, "Total", "black")
                )

            month_starts = []
            month_labels = []
            if show_month_ticks:
                month_start_days = [
                    0, 31, 59, 90, 120, 151,
                    181, 212, 243, 273, 304, 334
                ]

                for day in range(timespan + 1):
                    if day % 365 in month_start_days:
                        month_starts.append(day)
                        month_labels.append(
                            calendar.month_abbr[
                                month_start_days.index(day % 365) + 1
                            ]
                        )

            fig = plt.figure(figsize=(16, 4 * len(graph_data)))

            for index, (adults, babies, label, color) in enumerate(graph_data):
                ax = fig.add_subplot(len(graph_data), 1, index + 1)

                if separate_by_life_stage:
                    total = [baby + adult for baby, adult in zip(babies, adults)] # type: ignore

                    ax.plot(
                        days, babies, label=f"{label} babies", # type: ignore
                        color=color, linestyle=":", linewidth=2
                    )
                    ax.plot(
                        days, adults, label=f"{label} adults",
                        color=color, linestyle="--", linewidth=2
                    )
                    ax.plot(
                        days, total, label=f"{label} population",
                        color=color, linestyle="-", linewidth=2.5
                    )
                else:
                    ax.plot(
                        days, adults, label=f"{label} population",
                        color=color, linewidth=2.5
                    )

                ax.set_title(f"{label} Population", loc="left")
                ax.set_xlabel("Time (days)")
                ax.set_ylabel("Population")

                # Put the legend above the plotting area.
                ax.legend(
                    loc="lower right",
                    bbox_to_anchor=(1, 1.02),
                    ncol=3,
                    frameon=False,
                    borderaxespad=0
                )

                if show_month_ticks:
                    ax.set_xticks(
                        month_starts, month_labels, rotation=45
                    )

                if last_year_only:
                    ax.set_xlim(
                        days[max(0, timespan - 365)], timespan
                    )

            plt.tight_layout()
            if save_path:
                fig.savefig(save_path)

            plt.show()
            return

        # separate plots if we are separated by lifespan
        if not separate_by_life_stage:
            plt.plot(days, female_population, label='Female Population', color='red', linewidth=2.5)
            plt.plot(days, male_population, label='Male Population', color='blue', linewidth=2.5)
            if plot_total:
                total_population = [female + male for female, male in zip(female_population, male_population)]
                plt.plot(days, total_population, label='Total Population', color='black', linewidth=2.5)
        else:
            # first check to make sure we actually have baby data
            if baby_female_population is None or baby_male_population is None:
                raise ValueError("If separate_by_life_stage is True, baby_female_population and baby_male_population must be provided.")

            # print baby and adult populations for both
            plt.plot(days, baby_female_population, label='Baby Female Population', color='red', linewidth=2, linestyle='--')
            plt.plot(days, female_population, label='Adult Female Population', color='red', linewidth=2, linestyle=':')
            plt.plot(days, baby_male_population, label='Baby Male Population', color='blue', linewidth=2, linestyle='--')
            plt.plot(days, male_population, label='Adult Male Population', color='blue', linewidth=2, linestyle=':')
            # if we are plotting totals, plot total female, total male, and total overall
            if plot_total:
                total_female_population = [baby + adult for baby, adult in zip(baby_female_population, female_population)]
                total_male_population = [baby + adult for baby, adult in zip(baby_male_population, male_population)]
                total_population = [female + male for female, male in zip(total_female_population, total_male_population)]

                plt.plot(days, total_female_population, label='Total Female Population', color='red', linewidth=2.5)
                plt.plot(days, total_male_population, label='Total Male Population', color='blue', linewidth=2.5)
                plt.plot(days, total_population, label='Total Population', color='black', linewidth=2.5)

        plt.xlabel('Time (days)')
        plt.ylabel('Population (Bees)')
        plt.title('Bee Population Over Time')
        plt.legend()

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

        if last_year_only:
            plt.xlim(days[max(0, timespan - 365)], timespan) # in case there isn't a year of data
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        return

    def get_egg_laying_rate(self, t: float, zero_point: int = 300) -> float:
        """Get the egg laying rate at a given time

        :param t: The time since january 1st in days
        :type t: float
        :return: The egg laying rate (per day)
        :rtype: float
        """
        anchor_days = [0, 15, 74, 105, 166, 197, 228, zero_point, 350]
        anchor_rates = [41.67, 83.3, 0, 400, 1800, 1600, 1300, 0, 0]
        return np.interp(t % 365, anchor_days, anchor_rates, period=365) * self.EGG_LAYING_MULTIPLIER

    def get_female_means(self, t: float) -> float:
        """Get the mean lifespan for a female bee at the specified time

        :param t: The time since January 1st in days
        :type t: float
        :return: The mean lifespan for a female bee at the specified time
        :rtype: float
        """
        anchor_means = [150, 150, 150, 22.37, 19.29, 23.84, 22.84, 13.58, 26.25, 150, 150, 150]
        anchor_points = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        return np.interp(t % 365, anchor_points, anchor_means, period=365)

    def get_female_stds(self, t: float) -> float:
        """Get the standard deviation in lifespan for a female bee at the specified time

        :param t: The time since January 1st in days
        :type t: float
        :return: The standard deviation in lifespan for a female bee at the specified time
        :rtype: float
        """
        anchor_stds = [30, 30, 30, 1.89, 4.64, 3.91, 3.6, 2.06, 5.08, 30, 30, 30]
        anchor_points = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        return np.interp(t % 365, anchor_points, anchor_stds, period=365)

    def day_to_month(self, t: float) -> int:
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

    def find_weibull_survival_probability(self, age: float, mean: float, std: float) -> float:
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
        # include this so that we don't double count deaths during the growth period since we already account for this
        # in the transform_life_stage_population function
        exponent = (std / mean) ** -1.086
        lambda_param = mean / math.gamma(1 + (1 / exponent))
        return np.exp(-(age / lambda_param) ** exponent)

    def find_life_stage_population(self, egg_history: np.ndarray, growth_duration: int, gender: str = 'female', growth_success_rate: float = 0.85) -> tuple[list[float], list[float]]:
        """Find the bee population over time, separated into babies and adults

        :param egg_history: The number of eggs laid per day over the time period
        :type egg_history: np.ndarray
        :param growth_duration: How long it takes a baby bee to grow up
        :type growth_duration: int
        :param gender: The gender of bee to track the population of, defaults to 'female'
        :type gender: str, optional
        :param growth_success_rate: The chance a bee survives to adulthood, defaults to 0.85
        :type growth_success_rate: float, optional
        :return: Two lists containing the population over time. The first list is baby population, second list is adult population
        :rtype: tuple[list[float], list[float]]
        """
        # begin by calculating daily survival rate
        daily_survival_rate = growth_success_rate ** (1 / growth_duration)
        
        brood_by_age = np.zeros(growth_duration)
        baby_population_history = np.zeros(len(egg_history))
        adult_population_history = np.zeros(len(egg_history))

        for day, eggs_today in enumerate(egg_history):
            new_adults = brood_by_age[-1] * daily_survival_rate
            brood_by_age[1:] = brood_by_age[:-1] * daily_survival_rate
            brood_by_age[0] = eggs_today
            baby_population_history[day] = brood_by_age.sum()

            if new_adults > 0:
                if gender == 'female':
                    mean = self.get_female_means(day)
                    std = self.get_female_stds(day)
                else:
                    mean, std = 26.5, 5.5

                for current_day in range(day, len(egg_history)):
                    adult_age = current_day - day

                    adult_population_history[current_day] += (
                        new_adults
                        * self.find_weibull_survival_probability(
                            adult_age, mean, std
                        )
                    )

        return baby_population_history.tolist(), adult_population_history.tolist()

    def initial_population(self, t: int, starting_population: float) -> list[float]:
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
            population = starting_population * self.find_weibull_survival_probability(current_day, 150, 30)
            populations_history.append(population)
        return populations_history

    def main(self, graph: bool = True) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        """Run the simulation and optionally graph the results

        :param graph: Whether or not to graph the results, defaults to True
        :type graph: bool, optional
        :return: A tuple containing the population histories for each life stage (baby female, baby male, adult female, adult male, initial population)
        :rtype: tuple[list[float], list[float], list[float], list[float], list[float]]
        """
        # get the egg laying rate for the time period
        egg_history = np.array([
            self.get_egg_laying_rate(day, self.ZERO_DAY) for day in range(self.TIME_SPAN)
        ])

        # find female population and scale by FEMALE_PROPORTION to account for male:female ratio
        baby_female_population_history, adult_female_population_history = self.find_life_stage_population(egg_history, growth_duration=self.FEMALE_GROWTH_DURATION, gender='female', growth_success_rate=self.GROWTH_SUCCESS_RATE)
        baby_female_population_history = [female * self.FEMALE_PROPORTION for female in baby_female_population_history]
        adult_female_population_history = [female * self.FEMALE_PROPORTION for female in adult_female_population_history]

        # find male population and scale by MALE_PROPORTION to account for male:female ratio
        baby_male_population_history, adult_male_population_history = self.find_life_stage_population(egg_history, growth_duration=self.MALE_GROWTH_DURATION, gender='male', growth_success_rate=self.GROWTH_SUCCESS_RATE)
        baby_male_population_history = [male * self.MALE_PROPORTION for male in baby_male_population_history]
        adult_male_population_history = [male * self.MALE_PROPORTION for male in adult_male_population_history]

        # find initial population and how it falls off over time
        initial_population_history = self.initial_population(self.TIME_SPAN, self.INITIAL_POPULATION)
        
        # all initial bees are female, so add to the female population history
        adult_female_population_history = [female + initial for female, initial in zip(adult_female_population_history, initial_population_history)]

        # graph
        if graph:
            self.display_graph(
                adult_female_population_history,
                adult_male_population_history,
                self.TIME_SPAN,
                last_year_only=False,
                show_month_ticks=True,
                separate_by_life_stage=True,
                baby_female_population=baby_female_population_history,
                baby_male_population=baby_male_population_history,
                separate_graphs=True
            )

        return baby_female_population_history, baby_male_population_history, adult_female_population_history, adult_male_population_history, initial_population_history
