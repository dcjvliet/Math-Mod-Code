import calendar
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import PercentFormatter

from model_population import BeePopulationModel


class SensitivityAnalysis:
    def __init__(self,
                    time_span: int = 730,
                    initial_population: int = 5068,
                    female_proportion: float = 0.925,
                    male_proportion: float = 0.075,
                    female_growth_duration: int = 21,
                    male_growth_duration: int = 24,
                    growth_success_rate: float = 0.85,
                    egg_laying_multiplier: float = 1,
                    female_winter_lifespan: float = 150,
                    female_winter_standard_deviation: float = 30,
                    start_date: int = 1,
                ):
        """Initialize a SensitivityAnalysis object

        :param time_span: The number of days to model the population for (starting on Jan 1), defaults to 730
        :type time_span: int, optional
        :param initial_population: The initial population of the bee colony, defaults to 15000
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
        :param female_winter_lifespan: The average lifespan of a female bee during the winter, defaults to 150
        :type female_winter_lifespan: float, optional
        :param female_winter_standard_deviation: The standard deviation of the female bee lifespan during the winter, defaults to 30
        :type female_winter_standard_deviation: float, optional
        :param start_date: The starting day of the simulation, defaults to 1
        :type start_date: int, optional
        """
        self.TIME_SPAN = time_span # days
        self.INITIAL_POPULATION = initial_population
        self.FEMALE_PROPORTION = female_proportion
        self.MALE_PROPORTION = male_proportion
        self.FEMALE_GROWTH_DURATION = female_growth_duration
        self.MALE_GROWTH_DURATION = male_growth_duration
        self.GROWTH_SUCCESS_RATE = growth_success_rate
        self.EGG_LAYING_MULTIPLIER = egg_laying_multiplier
        self.FEMALE_WINTER_LIFESPAN = female_winter_lifespan
        self.FEMALE_WINTER_STANDARD_DEVIATION = female_winter_standard_deviation
        self.START_DATE = start_date

        plt.rcParams.update({
            'axes.titlesize': 20,     # Title size
            'axes.labelsize': 20,     # X and Y label size
            'xtick.labelsize': 16,    # X tick label size
            'ytick.labelsize': 16,    # Y tick label size
            'legend.fontsize': 14     # Legend font size
        })

    def vary_egg_laying_rate(self, variation_range: float = 0.1, graph: bool = False, trials: int = 101) -> tuple[list[float], list[float]]:
        """A function to model how peak population changes with a changing egg laying rate

        :param variation_range: The multiplier for the egg laying rate (will test 1 +- variation_range), defaults to 0.1
        :type variation_range: float, optional
        :param graph: Whether or not to graph the results, defaults to False
        :type graph: bool, optional
        :param trials: The number of trials to run (adds 1 if not odd), defaults to 101
        :type trials: int, optional
        :return: A tuple containing a list of peak populations from each trial, and a list of standard deviations for each trial
        :rtype: tuple[list[float], list[float]]
        """
        # we want to see how the population changes if our egg laying rate is slightly off
        # we will use peak population as a measure for each model
        if trials % 2 == 0:
                trials += 1 # make sure it's odd so there is one with 0 variation from assumed

        percentage_variations = []
        stdevs = []
        # get base population history for percentage comparisons
        base_population_history = []
        base_model = BeePopulationModel(
            time_span=self.TIME_SPAN,
            initial_population=self.INITIAL_POPULATION,
            female_proportion=self.FEMALE_PROPORTION,
            male_proportion=self.MALE_PROPORTION,
            female_growth_duration=self.FEMALE_GROWTH_DURATION,
            male_growth_duration=self.MALE_GROWTH_DURATION,
            growth_success_rate=self.GROWTH_SUCCESS_RATE,
            egg_laying_multiplier=1,
            female_winter_lifespan=self.FEMALE_WINTER_LIFESPAN,
            female_winter_standard_deviation=self.FEMALE_WINTER_STANDARD_DEVIATION,
            start_date=self.START_DATE
        )
        bf, bm, af, am, _ = base_model.main(graph=False)
        base_population_history = [b + a + m + j for b, a, m, j in zip(bf, af, bm, am)]

        for i in range(trials):
            multiplier = 1 + (i - (trials // 2)) * (variation_range / (trials // 2))
            # create a new model with new egg laying rate
            model = BeePopulationModel(
                time_span=self.TIME_SPAN,
                initial_population=self.INITIAL_POPULATION,
                female_proportion=self.FEMALE_PROPORTION,
                male_proportion=self.MALE_PROPORTION,
                female_growth_duration=self.FEMALE_GROWTH_DURATION,
                male_growth_duration=self.MALE_GROWTH_DURATION,
                growth_success_rate=self.GROWTH_SUCCESS_RATE,
                egg_laying_multiplier=multiplier,
                female_winter_lifespan=self.FEMALE_WINTER_LIFESPAN,
                female_winter_standard_deviation=self.FEMALE_WINTER_STANDARD_DEVIATION,
                start_date=self.START_DATE
            )
            bf, bm, af, am, _ = model.main(graph=False)
            # initial is already accounted for in the female population
            total_population = [b + a + m + j for b, a, m, j in zip(bf, af, bm, am)]
            trial_variations = []
            for i, pop in enumerate(total_population):
                trial_variations.append(pop / base_population_history[i])
            percentage_variations.append(sum(trial_variations) / len(trial_variations))

            for i, variation in enumerate(trial_variations):
                trial_variations[i] = variation - 1 # scale to percentage variation for stdev calc
            stdevs.append(np.std(trial_variations))

        # calculate its difference from a line
        x = np.array([1 + (i - (trials // 2)) * (variation_range / (trials // 2)) for i in range(trials)]) * 100
        y = (np.array(percentage_variations) * 100) - 100 # make percents and then make variations
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        predicted_y = m * x + c
        sum_residuals_squared = np.sum((y - predicted_y) ** 2)
        mean = np.mean(y)
        sum_distance_squared = np.sum((y - mean) ** 2)
        r_squared = 1 - (sum_residuals_squared / sum_distance_squared)

        if graph:
            annotation_x = x[int(len(x) / 1.5)]
            plt.annotate(
                f"y = {m:.2f}x {'+' if c >= 0 else '-'} {abs(c):.2f}",
                xy=(annotation_x, m * annotation_x + c),
                xytext=(0.65, 0.4),
                textcoords="axes fraction",
                fontsize=16,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="blue"),
                arrowprops=dict(arrowstyle="-", color="blue"),
            )

            plt.plot(x, y, color='green', linewidth=2.5, label='Simulation Data')
            plt.xlabel('Percentage of Original Egg Laying Rate')
            plt.ylabel('Average Percentage Variation of Population')
            plt.title('Average Percentage Variation of Population vs Percentage of Original Egg Laying Rate')
            # plt.text((1 - variation_range * 0.75) * 100, (max(y) - min(y)) * 0.90 + min(y), f'Error (log(1-R²)) ≈ {np.log(1-r_squared):.1f}', fontsize=16, color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.75))
            plt.grid()
            plt.legend()
            plt.show()

        return percentage_variations, stdevs

    def vary_growth_success_rate(self, variation_range: float = 0.1, graph: bool = False, trials: int = 101) -> tuple[list[float], list[float]]:
        """A function to model how peak population changes with chance of survival to adulthood

        :param variation_range: The multiplier for the growth success rate (will test 1 +- variation_range), defaults to 0.1
        :type variation_range: float, optional
        :param graph: Whether or not to graph the results, defaults to False
        :type graph: bool, optional
        :param trials: The number of trials to run (adds 1 if odd), defaults to 101
        :type trials: int, optional
        :return: A tuple containing the error from a lienar fit (NMSE, normalized with mean squared) and a list of peak populations from each trial
        :rtype: tuple[float, list[float]]
        """
        # we want to see how the population changes if our growth success rate is slightly off
        # we will use peak population as a measure for each model
        percent_differences = []
        stdevs = []
        if trials % 2 == 0:
            trials += 1

        base_population_history = []
        base_model = BeePopulationModel(
            time_span=self.TIME_SPAN,
            initial_population=self.INITIAL_POPULATION,
            female_proportion=self.FEMALE_PROPORTION,
            male_proportion=self.MALE_PROPORTION,
            female_growth_duration=self.FEMALE_GROWTH_DURATION,
            male_growth_duration=self.MALE_GROWTH_DURATION,
            growth_success_rate=self.GROWTH_SUCCESS_RATE,
            egg_laying_multiplier=1,
            female_winter_lifespan=self.FEMALE_WINTER_LIFESPAN,
            female_winter_standard_deviation=self.FEMALE_WINTER_STANDARD_DEVIATION,
            start_date=self.START_DATE
        )
        bf, bm, af, am, _ = base_model.main(graph=False)
        base_population_history = [b + a + m + j for b, a, m, j in zip(bf, af, bm, am)]

        for i in range(trials):
            new_success_rate = (1 + (i - (trials // 2)) * (variation_range / (trials // 2))) * self.GROWTH_SUCCESS_RATE
            # create a new model with new growth success rate
            model = BeePopulationModel(
                time_span=self.TIME_SPAN,
                initial_population=self.INITIAL_POPULATION,
                female_proportion=self.FEMALE_PROPORTION,
                male_proportion=self.MALE_PROPORTION,
                female_growth_duration=self.FEMALE_GROWTH_DURATION,
                male_growth_duration=self.MALE_GROWTH_DURATION,
                growth_success_rate=new_success_rate,
                egg_laying_multiplier=self.EGG_LAYING_MULTIPLIER,
                female_winter_lifespan=self.FEMALE_WINTER_LIFESPAN,
                female_winter_standard_deviation=self.FEMALE_WINTER_STANDARD_DEVIATION,
                start_date=self.START_DATE
            )
            bf, bm, af, am, _ = model.main(graph=False)
            total_population = [b + a + m + j for b, a, m, j in zip(bf, af, bm, am)]
            trial_variations = []
            for i, pop in enumerate(total_population):
                trial_variations.append(pop / base_population_history[i])
            percent_differences.append(sum(trial_variations) / len(trial_variations))

            for i, variation in enumerate(trial_variations):
                trial_variations[i] = variation - 1 # scale to percentage variation for stdev calc
            stdevs.append(np.std(trial_variations))

        # calculate its difference from a line
        x = np.array([(1 + (i - (trials // 2)) * (variation_range / (trials // 2))) * self.GROWTH_SUCCESS_RATE for i in range(trials)]) * 100 # make percent
        y = (np.array(percent_differences) * 100) - 100
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        predicted_y = m * x + c
        sum_residuals_squared = np.sum((y - predicted_y) ** 2)
        mean = np.mean(y)
        sum_distance_squared = np.sum((y - mean) ** 2)
        r_squared = 1 - (sum_residuals_squared / sum_distance_squared)

        if graph:
            annotation_x = x[int(len(x) / 1.5)]
            plt.annotate(
                f"y = {m:.2f}x {'+' if c >= 0 else '-'} {abs(c):.2f}",
                xy=(annotation_x, m * annotation_x + c),
                xytext=(0.65, 0.4),
                textcoords="axes fraction",
                fontsize=16,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="blue"),
                arrowprops=dict(arrowstyle="-", color="blue"),
            )

            plt.plot(x, y, color='green', linewidth=2.5, label='Simulation Data')
            plt.xlabel('Growth Success Rate (%)')
            plt.ylabel('Average Percentage Variation of Population (Bees)')
            plt.title('Average Percentage Variation of Population vs Growth Success Rate')
            plt.text(100 * (self.GROWTH_SUCCESS_RATE * (1 - variation_range * 0.75)), (max(y) - min(y)) * 0.90 + min(y), f'Error (log(1-R²)):\n ≈{float(np.log(1 - r_squared)):.2f}', fontsize=16, color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.75))
            plt.grid()
            plt.legend()
            plt.show()

        return percent_differences, stdevs

    def vary_male_female_proportion(self, lowest_male_proportion: float = 0.025, highest_male_proportion: float = 0.15, graph: bool = False, trials: int = 100) -> list[list[float]]:
        """A function to see how population is affected by the male vs female proportion

        :param lowest_male_proportion: The lowest proportion of males to test, defaults to 0.025
        :type lowest_male_proportion: float, optional
        :param highest_male_proportion: The highest proportion of males to test, defaults to 0.15
        :type highest_male_proportion: float, optional
        :param graph: Whether or not to graph the results, defaults to False
        :type graph: bool, optional
        :param trials: The number of trials to run, defaults to 100
        :type trials: int, optional
        :return: A list of lists containing the total population over time for each trial
        :rtype: list[list[float]]
        """
        population_histories = []

        for male_proportion in np.linspace(lowest_male_proportion, highest_male_proportion, trials):
            model = BeePopulationModel(
                time_span=self.TIME_SPAN,
                initial_population=self.INITIAL_POPULATION,
                female_proportion=1 - male_proportion,
                male_proportion=male_proportion,
                female_growth_duration=self.FEMALE_GROWTH_DURATION,
                male_growth_duration=self.MALE_GROWTH_DURATION,
                growth_success_rate=self.GROWTH_SUCCESS_RATE,
                egg_laying_multiplier=self.EGG_LAYING_MULTIPLIER,
                female_winter_lifespan=self.FEMALE_WINTER_LIFESPAN,
                female_winter_standard_deviation=self.FEMALE_WINTER_STANDARD_DEVIATION,
                start_date=self.START_DATE
            )

            bf, bm, af, am, _ = model.main(graph=False)
            total_population = np.sum(np.asarray([bf, bm, af, am], dtype=float), axis=0).tolist()
            population_histories.append(total_population)

        if graph:
            fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

            cmap = plt.get_cmap("coolwarm")
            norm = Normalize(vmin=lowest_male_proportion, vmax=highest_male_proportion)

            for male_proportion, population_history in zip(np.linspace(lowest_male_proportion, highest_male_proportion, trials), population_histories):
                ax.plot(
                    np.arange(self.START_DATE - 1, self.START_DATE - 1 + len(population_history)),
                    population_history,
                    color=cmap(norm(male_proportion)),
                    linewidth=1
                )

            colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
            colorbar.set_label("Proportion male")
            colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

            ax.set_title("Bee Population Over Time by Male and Female Proportions")
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Total Population (bees)")
            ax.grid(True, alpha=0.25)

            month_names = list(calendar.month_abbr)[1:]
            month_starts = []
            month_labels = []
            month_start_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

            for day in range(self.START_DATE - 1, self.START_DATE + self.TIME_SPAN):
                year_day = day % 365

                if year_day in month_start_days:
                    month_starts.append(day)
                    month_labels.append(month_names[month_start_days.index(year_day)])

            plt.xticks(month_starts, month_labels, rotation=45)

            plt.show()

        return population_histories

    def vary_female_lifespan(self, variation_range: float = 0.1, graph: bool = False, trials: int = 101) -> list[list[float]]:
        """A function to see how female population changes with female winter lifespan

        :param variation_range: The multiplier for female lifespan (will be 1 +- variation_range), defaults to 0.1
        :type variation_range: float, optional
        :param graph: Whether or not to graph the results, defaults to False
        :type graph: bool, optional
        :param trials: The number of trials to run (adds 1 if even), defaults to 101
        :type trials: int, optional
        :return: A list of population histories for each trial, where each population history is a list of female populations over time
        :rtype: list[list[float]]
        """
        population_histories = []

        if trials % 2 == 0:
            trials += 1

        for i in range(trials):
            new_lifespan = (1 + (i - (trials // 2)) * (variation_range / (trials // 2))) * self.FEMALE_WINTER_LIFESPAN
            new_standard_dev = new_lifespan * (self.FEMALE_WINTER_STANDARD_DEVIATION / self.FEMALE_WINTER_LIFESPAN)

            model = BeePopulationModel(
                time_span=self.TIME_SPAN,
                initial_population=self.INITIAL_POPULATION,
                female_proportion=self.FEMALE_PROPORTION,
                male_proportion=self.MALE_PROPORTION,
                female_growth_duration=self.FEMALE_GROWTH_DURATION,
                male_growth_duration=self.MALE_GROWTH_DURATION,
                growth_success_rate=self.GROWTH_SUCCESS_RATE,
                egg_laying_multiplier=self.EGG_LAYING_MULTIPLIER,
                female_winter_lifespan=new_lifespan,
                female_winter_standard_deviation=new_standard_dev,
                start_date=self.START_DATE
            )

            bf, _, af, _, _ = model.main(graph=False)
            total_population_f = np.sum(np.asarray([bf, af], dtype=float), axis=0).tolist()
            population_histories.append(total_population_f)

        if graph:
            fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

            lifespans = np.linspace(
                (1 - variation_range) * self.FEMALE_WINTER_LIFESPAN,
                (1 + variation_range) * self.FEMALE_WINTER_LIFESPAN,
                trials,
            )

            cmap = plt.get_cmap("coolwarm")
            norm = Normalize(vmin=min(lifespans), vmax=max(lifespans))

            for lifespan, population_history in zip(
                lifespans, population_histories
            ):
                ax.plot(
                    np.arange(self.START_DATE - 1, self.START_DATE - 1 + len(population_history)),
                    population_history,
                    color=cmap(norm(lifespan)),
                    linewidth=1,
                )

            colorbar = fig.colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                ax=ax,
            )

            colorbar.ax.yaxis.set_ticks_position('left')
            colorbar.ax.yaxis.set_label_position('left')
            colorbar.set_label('Mean female winter lifespan (days)')

            std_ratio = (
                self.FEMALE_WINTER_STANDARD_DEVIATION
                / self.FEMALE_WINTER_LIFESPAN
            )

            std_axis = colorbar.ax.secondary_yaxis(
                'right',
                functions=(
                    lambda mean: np.asarray(mean, dtype=float) * std_ratio,
                    lambda std: np.asarray(std, dtype=float) / std_ratio,
                ),
            )
            std_axis.set_ylabel('Female winter lifespan standard deviation (days)')

            ax.set_title("Female Population Over Time by Female Winter Lifespan")
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Female Population (Bees)")
            ax.grid(True, alpha=0.25)

            month_names = list(calendar.month_abbr)[1:]
            month_starts = []
            month_labels = []
            month_start_days = [
                0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334
            ]

            for day in range(self.START_DATE - 1, self.START_DATE + self.TIME_SPAN):
                year_day = day % 365

                if year_day in month_start_days:
                    month_starts.append(day)
                    month_labels.append(
                        month_names[month_start_days.index(year_day)]
                    )

            ax.set_xticks(month_starts)
            ax.set_xticklabels(month_labels, rotation=45)

            plt.show()

        return population_histories

    def vary_female_standard_deviation(self, lowest_female_standard_deviation: float = 10, highest_female_standard_deviation: float = 30, graph: bool = False, trials: int = 101) -> list[list[float]]:
        """A function to see how population is affected by the standard deviation of female winter lifespan

        :param lowest_female_standard_deviation: The lowest standard deviation to test (in days), defaults to 10
        :type lowest_female_standard_deviation: float, optional
        :param highest_female_standard_deviation: The highest standard deviation to test (in days), defaults to 30
        :type highest_female_standard_deviation: float, optional
        :param graph: Whether to generate a graph of the results, defaults to False
        :type graph: bool, optional
        :param trials: The number of trials to run (adds 1 if even), defaults to 101
        :type trials: int, optional
        :return: A list of population histories for each trial
        :rtype: list[list[float]]
        """
        population_histories = []

        for female_standard_deviation in np.linspace(lowest_female_standard_deviation, highest_female_standard_deviation, trials):
            model = BeePopulationModel(
                time_span=self.TIME_SPAN,
                initial_population=self.INITIAL_POPULATION,
                female_proportion=0.925,
                male_proportion=0.075,
                female_growth_duration=self.FEMALE_GROWTH_DURATION,
                male_growth_duration=self.MALE_GROWTH_DURATION,
                growth_success_rate=self.GROWTH_SUCCESS_RATE,
                egg_laying_multiplier=self.EGG_LAYING_MULTIPLIER,
                female_winter_lifespan=self.FEMALE_WINTER_LIFESPAN,
                female_winter_standard_deviation=female_standard_deviation,
                start_date=self.START_DATE
            )

            bf, bm, af, am, _ = model.main(graph=False)
            total_population = np.sum(np.asarray([bf, bm, af, am], dtype=float), axis=0).tolist()
            population_histories.append(total_population)

        if graph:
            fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

            cmap = plt.get_cmap("coolwarm")
            norm = Normalize(vmin=lowest_female_standard_deviation, vmax=highest_female_standard_deviation)

            for female_standard_deviation, population_history in zip(np.linspace(lowest_female_standard_deviation, highest_female_standard_deviation, trials), population_histories):
                ax.plot(
                    np.arange(self.START_DATE - 1, self.START_DATE - 1 + len(population_history)),
                    population_history,
                    color=cmap(norm(female_standard_deviation)),
                    linewidth=1
                )

            colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
            colorbar.set_label("Female Winter Lifespan Standard Deviation")

            ax.set_title("Bee Population Over Time by Female Winter Lifespan Standard Deviation")
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Total Population (bees)")
            ax.grid(True, alpha=0.25)

            month_names = list(calendar.month_abbr)[1:]
            month_starts = []
            month_labels = []
            month_start_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

            for day in range(self.START_DATE - 1, self.START_DATE + self.TIME_SPAN):
                year_day = day % 365

                if year_day in month_start_days:
                    month_starts.append(day)
                    month_labels.append(month_names[month_start_days.index(year_day)])

            plt.xticks(month_starts, month_labels, rotation=45)

            plt.show()

        return population_histories
    