from matplotlib import pyplot as plt
import numpy as np


WIDTH = 200
LENGTH = 400
STEP_SIZE = 0.1
GRID = [[0 for _ in range(WIDTH)] for _ in range(LENGTH)]
NUM_NOZZLES = 19
NOZZLE_RADIUS = 25

WIND_STEP_SIZE = 0.1
WIND_MIN = 0
WIND_MAX = 15


def chord_length(distance):
    return 2 * (NOZZLE_RADIUS ** 2 - distance ** 2) ** 0.5


def std(distribution):
    mean = sum(distribution) / len(distribution)
    variance = sum((x - mean) ** 2 for x in distribution) / len(distribution)
    return variance ** 0.5

std_data = []
for speed in range(int((WIND_MAX - WIND_MIN) / WIND_STEP_SIZE + 1)):
    wind_speed = WIND_MIN + speed * WIND_STEP_SIZE

    NOZZLE_CENTERS = [220 / (NUM_NOZZLES - 1) * i - (220 - 200) / 2 + wind_speed * 0.788 for i in range(NUM_NOZZLES)]
    distribution = []
    for i in range(round(WIDTH / STEP_SIZE)):
        x_cord = i * STEP_SIZE
        total_water = 0
        for center in NOZZLE_CENTERS:
            distance = abs(x_cord - center)
            if distance <= NOZZLE_RADIUS:
                total_water += chord_length(distance) * (1 - distance / (2 * NOZZLE_RADIUS))

        distribution.append(total_water)

    normalized_distribution = [x / max(distribution) for x in distribution]

    std_data.append(std(normalized_distribution))

plt.plot([WIND_MIN + speed * WIND_STEP_SIZE for speed in range(int((WIND_MAX - WIND_MIN) / WIND_STEP_SIZE + 1))], std_data)
plt.xlabel("Wind Speed (ft/s)")
plt.ylabel("Standard Deviation of Water Distribution")
plt.title("Effect of Wind on Water Distribution")
plt.show()
