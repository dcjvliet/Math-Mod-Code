import numpy as np


mean_aa_antenna_length = 1.514
std_aa_antenna_length = 0.096

mean_ax_antenna_length = 1.307
std_ax_antenna_length = 0.105

mean_antenna_length = (mean_aa_antenna_length + mean_ax_antenna_length) / 2

correct_count = 0
incorrect_count = 0
rng = np.random.default_rng()
aa_antenna_length = rng.normal(loc=mean_aa_antenna_length, scale=std_aa_antenna_length, size=1000000)
ax_antenna_length = rng.normal(loc=mean_ax_antenna_length, scale=std_ax_antenna_length, size=1000000)
for antenna_length in aa_antenna_length:
    if antenna_length > mean_antenna_length:
        correct_count += 1
    else:
        incorrect_count += 1

for antenna_length in ax_antenna_length:
    if antenna_length < mean_antenna_length:
        correct_count += 1
    else:
        incorrect_count += 1

print(f"Correct: {correct_count}, Incorrect: {incorrect_count}")
print(f"Accuracy: {correct_count / (correct_count + incorrect_count) * 100:.2f}%")