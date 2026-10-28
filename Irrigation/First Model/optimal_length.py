from matplotlib import pyplot as plt


WIDTH = 200
LENGTH = 400
STEP_SIZE = 0.1
GRID = [[0 for _ in range(WIDTH)] for _ in range(LENGTH)]
NUM_NOZZLES = 5
NOZZLE_RADIUS = 25


def chord_length(distance):
    return 2 * (NOZZLE_RADIUS ** 2 - distance ** 2) ** 0.5


def std(distribution):
    mean = sum(distribution) / len(distribution)
    variance = sum((x - mean) ** 2 for x in distribution) / len(distribution)
    return variance ** 0.5


minimum_stds = []
for pipe_length in range(150, 221):
    stds = []
    for nozzles in range(NUM_NOZZLES, 20):
        NOZZLE_CENTERS = [pipe_length / (nozzles - 1) * i - (pipe_length - 200) / 2 for i in range(nozzles)]
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
    minimum_stds.append(min(stds))

x = range(150, 221)

plt.plot(x, minimum_stds, marker='o')

plt.xlabel("Length of Pipe (ft)")
plt.ylabel("Standard Deviation in Water Distribution (uniformity)")
plt.xticks(x, rotation='vertical')
plt.grid(True)

plt.show()

print(minimum_stds[-2])
print(minimum_stds[-1])


# 219ft with 15 nozzles is optimal