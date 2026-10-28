from matplotlib import pyplot as plt
import numpy as np


WIDTH = 200
LENGTH = 400
STEP_SIZE = 0.1
GRID = [[0 for _ in range(WIDTH)] for _ in range(LENGTH)]
NUM_NOZZLES = 15
NOZZLE_RADIUS = 25


def chord_length(distance):
    return 2 * (NOZZLE_RADIUS ** 2 - distance ** 2) ** 0.5


def std(distribution):
    mean = sum(distribution) / len(distribution)
    variance = sum((x - mean) ** 2 for x in distribution) / len(distribution)
    return variance ** 0.5

NOZZLE_CENTERS = [219 / (NUM_NOZZLES - 1) * i - (219 - 200) / 2 for i in range(NUM_NOZZLES)]
distribution = []
for i in range(round(WIDTH / STEP_SIZE)):
    x_cord = i * STEP_SIZE
    total_water = 0
    for center in NOZZLE_CENTERS:
        distance = abs(x_cord - center)
        if distance <= NOZZLE_RADIUS:
            total_water += chord_length(distance)

    distribution.append(total_water)

print("Standard Deviation of Distribution:", std(distribution))
data = np.tile(distribution, (400, 1))

plt.figure(figsize=(10, 5))

plt.imshow(
    data,
    extent=(0, 200, 0, LENGTH),
    origin="lower",
    aspect="auto",
    cmap="coolwarm",  # blue = low, red = high
    vmin=min(distribution),
    vmax=max(distribution)
)

plt.xlabel("Width of field (ft)")
plt.ylabel("Length of field (ft)")
plt.xlim(0, 200)
plt.ylim(0, LENGTH)
plt.colorbar(label="Amount of Water (arbitrary units)")

plt.show()