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
                    initial_population: int = 6000,
                    female_proportion: float = 0.925,
                    male_proportion: float = 0.075,
                    female_growth_duration: int = 21,
                    male_growth_duration: int = 24,
                    growth_success_rate: float = 0.85,
                    egg_laying_multiplier: float = 1,
                    zero_day: int = 300,
                    female_winter_lifespan: float = 150
                ):
        """Initialize a SensitivityAnalysis object

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
        :param female_winter_lifespan: The average lifespan of a female bee during the winter, defaults to 150
        :type female_winter_lifespan: float, optional
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
        self.FEMALE_WINTER_LIFESPAN = female_winter_lifespan

        plt.rcParams.update({
            'axes.titlesize': 20,     # Title size
            'axes.labelsize': 20,     # X and Y label size
            'xtick.labelsize': 16,    # X tick label size
            'ytick.labelsize': 16,    # Y tick label size
            'legend.fontsize': 14     # Legend font size
        })

    def vary_egg_laying_rate(self, variation_range: float = 0.1, graph: bool = False, trials: int = 101) -> tuple[float, list[float]]:
        """A function to model how peak population changes with a changing egg laying rate

        :param variation_range: The multiplier for the egg laying rate (will test 1 +- variation_range), defaults to 0.1
        :type variation_range: float, optional
        :param graph: Whether or not to graph the results, defaults to False
        :type graph: bool, optional
        :param trials: The number of trials to run (adds 1 if not odd), defaults to 101
        :type trials: int, optional
        :return: A tuple containing the error from a linear fit (NMSE, normalized with mean) and a list of peak populations from each trial
        :rtype: tuple[float, list[float]]
        """
        # we want to see how the population changes if our egg laying rate is slightly off
        # we will use peak population as a measure for each model
        if trials % 2 == 0:
                trials += 1 # make sure it's odd so there is one with 0 variation from assumed

        peak_populations = []
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
                zero_day=self.ZERO_DAY,
                female_winter_lifespan=self.FEMALE_WINTER_LIFESPAN
            )
            bf, bm, af, am, _ = model.main(graph=False)
            # initial is already accounted for in the female population
            total_population = [b + a + m + j for b, a, m, j in zip(bf, af, bm, am)]
            peak_populations.append(max(total_population))

        # calculate its difference from a line
        x = np.array([1 + (i - (trials // 2)) * (variation_range / (trials // 2)) for i in range(trials)])
        y = np.array(peak_populations)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        error = 0
        for x_value, y_value in zip(x, y):
            error += abs(y_value - (m * x_value + c))

        error /= len(x)
        error /= sum(y) / len(y) # normalize by mean

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
            plt.xlabel('Egg Laying Rate Multiplier')
            plt.ylabel('Peak Population (Bees)')
            plt.title('Peak Population vs Egg Laying Rate Multiplier')
            plt.text(1 - variation_range * 0.75, (max(y) - min(y)) * 0.90 + min(y), f'Error from linear fit (NMSE):\n ≈{error:.2f}', fontsize=16, color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.75))
            plt.grid()
            plt.legend()
            plt.show()

        return error, peak_populations

    def vary_growth_success_rate(self, variation_range: float = 0.1, graph: bool = False, trials: int = 101) -> tuple[float, list[float]]:
        """A function to model how peak population changes with chance of survival to adulthood

        :param variation_range: The multiplier for the growth success rate (will test 1 +- variation_range), defaults to 0.1
        :type variation_range: float, optional
        :param graph: Whether or not to graph the results, defaults to False
        :type graph: bool, optional
        :param trials: The number of trials to run (adds 1 if odd), defaults to 101
        :type trials: int, optional
        :return: A tuple containing the error from a lienar fit (NMSE, normalized with mean) and a list of peak populations from each trial
        :rtype: tuple[float, list[float]]
        """
        # we want to see how the population changes if our growth success rate is slightly off
        # we will use peak population as a measure for each model
        peak_populations = []
        if trials % 2 == 0:
            trials += 1

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
                zero_day=self.ZERO_DAY,
                female_winter_lifespan=self.FEMALE_WINTER_LIFESPAN
            )
            bf, bm, af, am, _ = model.main(graph=False)
            total_population = [b + a + m + j for b, a, m, j in zip(bf, af, bm, am)]
            peak_populations.append(max(total_population))

        # calculate its difference from a line
        x = np.array([(1 + (i - (trials // 2)) * (variation_range / (trials // 2))) * self.GROWTH_SUCCESS_RATE for i in range(trials)])
        y = np.array(peak_populations)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        error = 0
        for x_value, y_value in zip(x, y):
            error += abs(y_value - (m * x_value + c))

        error /= len(x)
        error /= sum(y) / len(y) # normalize by mean

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
            plt.xlabel('Growth Success Rate')
            plt.ylabel('Peak Population (Bees)')
            plt.title('Peak Population vs Growth Success Rate')
            plt.text(0.85 * (1 - variation_range * 0.75), (max(y) - min(y)) * 0.90 + min(y), f'Error from linear fit (NMSE):\n ≈{error:.2f}', fontsize=16, color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.75))
            plt.grid()
            plt.legend()
            plt.show()

        return error, peak_populations

    def vary_egg_laying_rate_cutoff(self, graph: bool = False) -> tuple[float, list[float]]:
        """A function to see how average population changes with when the egg laying rate reaches 0 in the winter

        :param graph: Whether or not to graph the results, defaults to False
        :type graph: bool, optional
        :return: A tuple containing the error from a linear fit (NMSE, normalized with mean) and a list of average populations after the egg laying rate starts declining for each trial
        :rtype: tuple[float, list[float]]
        """
        average_populations = []
        for zero_day in range(229, 345): # we start decreasing at 229, and increase starting at 350, so we look at range in between
            model = BeePopulationModel(
                time_span=self.TIME_SPAN,
                initial_population=self.INITIAL_POPULATION,
                female_proportion=self.FEMALE_PROPORTION,
                male_proportion=self.MALE_PROPORTION,
                female_growth_duration=self.FEMALE_GROWTH_DURATION,
                male_growth_duration=self.MALE_GROWTH_DURATION,
                growth_success_rate=self.GROWTH_SUCCESS_RATE,
                egg_laying_multiplier=self.EGG_LAYING_MULTIPLIER,
                zero_day=zero_day,
                female_winter_lifespan=self.FEMALE_WINTER_LIFESPAN
            )
            bf, bm, af, am, _ = model.main(graph=False)
            total_population = [b + a + m + j for b, a, m, j in zip(bf, af, bm, am)]
            total_population = total_population[593:] # only look at population after egg laying rate starts declining until the end of the year
            average_populations.append(sum(total_population) / len(total_population)) # average population after egg laying rate starts declining

        # calculate its difference from a line
        x = np.array(range(229, 345))
        y = np.array(average_populations)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        error = 0
        for x_value, y_value in zip(x, y):
            error += abs(y_value - (m * x_value + c))

        error /= len(x)
        error /= sum(y) / len(y) # normalize by mean
        estimated_y = np.array([m * x_value + c for x_value in x])

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

            plt.plot(range(229, 345), average_populations, color='blue', linewidth=2.5, label='Simulation Data')
            plt.plot(x, estimated_y, color='green', linewidth=1.5, label='Linear Fit')
            plt.xlabel('Day Egg Laying Rate Hits Zero (Zero Day)')
            plt.ylabel('Average Population (After Egg Laying Rate Declines) (Bees)')
            plt.title('Average Population vs Zero Day')
            plt.legend()
            plt.text(245, (max(y) - min(y)) * 0.90 + min(y), f'Error from linear fit (NMSE):\n ≈{error:.2f}', fontsize=16, color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.75))
            plt.grid()

            # code for changing over to months
            month_names = list(calendar.month_abbr)[1:]
            month_starts = []
            month_labels = []
            month_start_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

            for day in range(229, 345):
                year_day = day % 365

                if year_day in month_start_days:
                    month_starts.append(day)
                    month_labels.append(month_names[month_start_days.index(year_day)])

            plt.xticks(month_starts, month_labels, rotation=45)

            plt.legend()
            plt.show()

        return error, average_populations

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
                zero_day=self.ZERO_DAY,
                female_winter_lifespan=self.FEMALE_WINTER_LIFESPAN
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
                    np.arange(len(population_history)),
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

            for day in range(self.TIME_SPAN + 1): # so it does january
                year_day = day % 365

                if year_day in month_start_days:
                    month_starts.append(day)
                    month_labels.append(month_names[month_start_days.index(year_day)])

            plt.xticks(month_starts, month_labels, rotation=45)

            plt.show()

        return population_histories

    def vary_growth_time(self, variation_range: float = 0.1, graph: bool = False) -> tuple[float, float, list[float], list[float]]:
        """A function to see how peak population changes with the growth time of the bees

        :param variation_range: The multiplier for the growth time (will be 1 +- variation_range), defaults to 0.1
        :type variation_range: float, optional
        :param graph: Whether or not to graph the results, defaults to False
        :type graph: bool, optional
        :return: A tuple containing the error from a linear fit (NMSE, normalized with mean) for female and male growth times, and lists of peak popualtions for female and male bees for each trial
        :rtype: tuple[float, float, list[float], list[float]]
        """
        # this one is weird because the growth time has to be an integer (i'm pretty sure)
        peak_populations_f = []
        peak_populations_m = []

        lower_int_f = math.floor(self.FEMALE_GROWTH_DURATION * (1 - variation_range)) 
        upper_int_f = math.ceil(self.FEMALE_GROWTH_DURATION * (1 + variation_range))
        lower_int_m = math.floor(self.MALE_GROWTH_DURATION * (1 - variation_range))
        upper_int_m = math.ceil(self.MALE_GROWTH_DURATION * (1 + variation_range))

        # we need to do it separately for male and female
        for female_growth_duration in range(lower_int_f, upper_int_f + 1):
            model = BeePopulationModel(
                time_span=self.TIME_SPAN,
                initial_population=self.INITIAL_POPULATION,
                female_proportion=self.FEMALE_PROPORTION,
                male_proportion=self.MALE_PROPORTION,
                female_growth_duration=female_growth_duration,
                male_growth_duration=self.MALE_GROWTH_DURATION,
                growth_success_rate=self.GROWTH_SUCCESS_RATE,
                egg_laying_multiplier=self.EGG_LAYING_MULTIPLIER,
                zero_day=self.ZERO_DAY,
                female_winter_lifespan=self.FEMALE_WINTER_LIFESPAN
            )
            bf, bm, af, am, _ = model.main(graph=False)
            total_population = [b + a + m + j for b, a, m, j in zip(bf, af, bm, am)]
            peak_populations_f.append(max(total_population))

        for male_growth_duration in range(lower_int_m, upper_int_m + 1):
            model = BeePopulationModel(
                time_span=self.TIME_SPAN,
                initial_population=self.INITIAL_POPULATION,
                female_proportion=self.FEMALE_PROPORTION,
                male_proportion=self.MALE_PROPORTION,
                female_growth_duration=self.FEMALE_GROWTH_DURATION,
                male_growth_duration=male_growth_duration,
                growth_success_rate=self.GROWTH_SUCCESS_RATE,
                egg_laying_multiplier=self.EGG_LAYING_MULTIPLIER,
                zero_day=self.ZERO_DAY,
                female_winter_lifespan=self.FEMALE_WINTER_LIFESPAN
            )
            bf, bm, af, am, _ = model.main(graph=False)
            total_population = [b + a + m + j for b, a, m, j in zip(bf, af, bm, am)]
            peak_populations_m.append(max(total_population))

        # now we have to calculate two regressions and graph separately
        x_f = np.array(range(lower_int_f, upper_int_f + 1))
        y_f = np.array(peak_populations_f)
        A_f = np.vstack([x_f, np.ones(len(x_f))]).T
        m_f, c_f = np.linalg.lstsq(A_f, y_f, rcond=None)[0]
        error_f = 0
        for x_value, y_value in zip(x_f, y_f):
            error_f += abs(y_value - (m_f * x_value + c_f))
        error_f /= len(x_f)
        error_f /= sum(y_f) / len(y_f) # normalize by mean

        x_m = np.array(range(lower_int_m, upper_int_m + 1))
        y_m = np.array(peak_populations_m)
        A_m = np.vstack([x_m, np.ones(len(x_m))]).T
        m_m, c_m = np.linalg.lstsq(A_m, y_m, rcond=None)[0]
        error_m = 0
        for x_value, y_value in zip(x_m, y_m):
            error_m += abs(y_value - (m_m * x_value + c_m))
        error_m /= len(x_m)
        error_m /= sum(y_m) / len(y_m) # normalize by mean

        if graph:
            fig, axes = plt.subplots(2, 1)

            for ax, x, y, m, c, peak_populations, error, color, population in [
                (axes[0], x_f, y_f, m_f, c_f, peak_populations_f, error_f, 'red', 'Female'),
                (axes[1], x_m, y_m, m_m, c_m, peak_populations_m, error_m, 'blue', 'Male'),
            ]:
                plt.sca(ax)

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

                plt.plot(
                    x, peak_populations,
                    color=color,
                    linewidth=4,
                    label=f'{population} Population Simulation Data',
                )
                plt.xlabel('Growth Duration (Days)')
                plt.ylabel('Peak Population (Bees)')
                plt.title(f'{population} Population: Peak Population vs Growth Duration')
                plt.text(
                    (x[1] + x[2]) / 2,
                    (max(y) - min(y)) * 0.84 + min(y),
                    f'Error from linear fit (NMSE):\n ≈{error:.2f}',
                    fontsize=16,
                    color='black',
                    ha='center',
                    va='center',
                    bbox=dict(facecolor='white', alpha=0.75),
                )
                plt.grid()
                plt.legend()

            fig.tight_layout()

            # match both margins to whichever needs more space
            margin = max(fig.subplotpars.left, 1 - fig.subplotpars.right)
            fig.subplots_adjust(left=margin, right=1 - margin)

            plt.show()

        return error_f, error_m, peak_populations_f, peak_populations_m

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

            model = BeePopulationModel(
                time_span=self.TIME_SPAN,
                initial_population=self.INITIAL_POPULATION,
                female_proportion=self.FEMALE_PROPORTION,
                male_proportion=self.MALE_PROPORTION,
                female_growth_duration=self.FEMALE_GROWTH_DURATION,
                male_growth_duration=self.MALE_GROWTH_DURATION,
                growth_success_rate=self.GROWTH_SUCCESS_RATE,
                egg_laying_multiplier=self.EGG_LAYING_MULTIPLIER,
                zero_day=self.ZERO_DAY,
                female_winter_lifespan=new_lifespan
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
                    np.arange(len(population_history)),
                    population_history,
                    color=cmap(norm(lifespan)),
                    linewidth=1,
                )

            colorbar = fig.colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                ax=ax,
            )
            colorbar.set_label("Female winter lifespan (days)")

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

            for day in range(self.TIME_SPAN + 1):
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
    