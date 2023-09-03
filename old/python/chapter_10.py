
# Load required packages

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

## Example: Gym trial conversions
# Implement the resampling procedure

random.seed(123)
box = [1, 0, 0, 0]
n_ones = []
for _ in range(1000):
    # sample 165 times with replacement
    n_ones.append(sum(random.choices(box, k=165)))
n_ones = np.array(n_ones)
count_above = sum(n_ones >= 53)
p_value = count_above / len(n_ones)
print(f"count above: {count_above}")
print(f"p-value: {p_value:.3f}")

# Plot the histogram

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(n_ones, bins=range(25, 61))
ax.axvline(53, color="black", linestyle="dashed")
ax.set_xlabel("Number of conversions")
ax.set_ylabel("Frequency")
plt.show()

## Example: Humidifier moisture output
# Load the dataset

data = pd.read_csv("vendors.txt", header=None)
vendorA = data[0].to_numpy()[:12]
vendorB = data[0].to_numpy()[12:]

### Resampling procedure for the confidence interval

rng = np.random.default_rng(seed=123)
box = list(vendorA)
means = []
for _ in range(1000):
    resample = rng.choice(box, 12, replace=True)
    means.append(np.mean(resample))
ci_interval = np.percentile(means, [5, 95])
print(ci_interval.round(2))

# Plot the histogram

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(means, bins=20)
ax.axvline(14, color="darkgrey")
ax.set_xlabel("Mean moisture output (oz./hr)")
ax.set_ylabel("Frequency")

for ci in ci_interval:
    ax.axvline(ci, color="black")
ax.plot(ci_interval, [180, 180], color="black", linestyle="--")
ax.text(ci_interval.mean(), 170, "90%-confidence interval", ha="center")
plt.show()

### Formula approach for the confidence interval

ci_interval = stats.t.interval(0.9, 11, loc=vendorA.mean(), scale=vendorA.std(ddof=1)/np.sqrt(12))
np.array(ci_interval).round(2)


observed = pd.DataFrame({
    "B": {1: 15, 2: 3, 3: 11},
    "I": {1: 24, 2: 5, 3: 1},
}).transpose()

# use scipy.stats.chi2_contingency to determine expected values
_, _, _, expected = stats.chi2_contingency(observed)
print(expected)
# calculate sum of absolute differences
np.abs(observed - expected).sum().sum()

## Chi Square Example

chi2, p_value, dof, expected = stats.chi2_contingency(observed)
stats.chi2_contingency(observed)

## Benford's law
# Following the equation in the Wikipedia article, we can

digits = np.arange(1, 10)
benford = pd.DataFrame({
    "digit": digits,
    "expected": np.log10(1 + 1/digits),
})
print(benford)
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(benford["digit"], benford["expected"])
ax.set_xticks(digits)
ax.set_xlabel("Leading Digit")
ax.set_ylabel("Expected Probability")
plt.show()

## Resampling distributions of interior digits
# Look at an example of a resampling distribution of interior digits.

from collections import Counter

random.seed(123)
box = list(range(10))
random.shuffle(box)
resample = random.choices(box, k=315)
counts = Counter(resample)
df = pd.DataFrame({
    "count": [counts[i] for i in range(10)],
}, index=range(10))

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(df.index, df["count"])
ax.axhline(31.5, color="black")
ax.set_xlabel("Digit")
ax.set_ylabel("Count")
plt.show()

# Resampling experiment

random.seed(123)
box = list(range(10))
differences = []
for _ in range(10_000):
    random.shuffle(box)
    resample = random.choices(box, k=315)
    counts = Counter(resample)
    df = pd.DataFrame({
        "count": [counts[i] for i in range(10)],
    }, index=range(10))
    differences.append(sum(abs(df["count"] - 31.5)))
differences = np.array(differences)
above_216 = sum(differences >= 216)
p_value = above_216 / len(differences)

print(f"Number of resamples with sum of absolute deviations >= 216: {above_216}")
print(f"p-value = {p_value:.4f}")
