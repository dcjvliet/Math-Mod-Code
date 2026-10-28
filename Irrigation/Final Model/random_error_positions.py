from matplotlib import pyplot as plt
import random


WIDTH = 200
LENGTH = 400
STEP_SIZE = 0.1
GRID = [[0 for _ in range(WIDTH)] for _ in range(LENGTH)]
NUM_NOZZLES = 19
NOZZLE_RADIUS = 25


def chord_length(distance):
    return 2 * (NOZZLE_RADIUS ** 2 - distance ** 2) ** 0.5


def std(distribution):
    mean = sum(distribution) / len(distribution)
    variance = sum((x - mean) ** 2 for x in distribution) / len(distribution)
    return variance ** 0.5


stds = []
for trial in range(100):
    NOZZLE_CENTERS = [220 / (NUM_NOZZLES - 1) * i - (220 - 200) / 2 + random.uniform(-0.25, 0.25) for i in range(NUM_NOZZLES)]
    distribution = []
    for i in range(round(WIDTH / STEP_SIZE)):
        x_cord = i * STEP_SIZE
        total_water = 0
        for center in NOZZLE_CENTERS:
            distance = abs(x_cord - center)
            if distance <= NOZZLE_RADIUS:
                total_water += chord_length(distance)

        distribution.append(total_water)

    normalized = [x / max(distribution) for x in distribution]
    stds.append(std(normalized))


std_of_stds = std(stds)
avg_of_stds = sum(stds) / len(stds)
print("Standard Deviation of Standard Deviations: ", std_of_stds)
print("Average of Standard Deviations: ", avg_of_stds)
x = range(100)

plt.plot(x, stds, marker='o')

plt.xlabel("Trial")
plt.ylabel("Standard Deviation in Water Distribution (uniformity)")
plt.grid(True)

plt.show()
