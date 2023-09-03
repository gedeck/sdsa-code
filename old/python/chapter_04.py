
# Load required packages

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("berkeley.csv")
df = df[df["Major"] != "Other"]

# All applicants for department A

subset = df[df["Major"] == "A"]
pd.crosstab(subset["Admission"], subset["Gender"], margins=True)

# All applicants for department A (variation)

subset = df[df["Major"] == "A"]
pd.crosstab(subset["Gender"], subset["Admission"], margins=True)

# All applicants for department E

subset = df[df["Major"] == "E"]
pd.crosstab(subset["Admission"], subset["Gender"], margins=True)

# All applicants for department E (percent by column)

(pd.crosstab(subset["Admission"], subset["Gender"],
             margins=True, normalize="columns") * 100).round(2)

### Resampling experiment

random.seed(123)
hat = [1] * 147 + [0] * 437

differences = []
for _ in range(2_000):
    random.shuffle(hat)
    females = hat[:393]
    males = hat[393:]
    # calculate difference and admission rates for females and males
    admit_females = 100 * sum(females) / len(females)
    admit_males = 100 * sum(males) / len(males)
    differences.append(admit_females - admit_males)

observed = -3.83
n_extreme = sum(np.array(differences) <= observed)
p_value = n_extreme / len(differences)
print(f"First trial difference {differences[0]:.2f}")
print(f"Number of extreme trials = {n_extreme}")
print(f"p_value = {p_value:.4f}")

# Visualize the distribution of differences

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(differences, bins=30, color="C0")
ax.axvline(observed, color="black", linestyle="--")
ax.set_xlabel("Difference in Admission Rates")
ax.set_ylabel("Frequency")
plt.show()

## Example: smoking and gender
# Load and process the data

data = pd.read_csv("PulseNew.csv")
data.head()

# Create 2x2 table

ct = pd.crosstab(data["Smokes?"], data["Sex"], margins=True, normalize="columns")
100 * ct.round(4)
