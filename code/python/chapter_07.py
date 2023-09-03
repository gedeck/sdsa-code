
# Load required packages

import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from scipy import stats

## Sampling Distribution for a Proportion
# Implement resampling simulation and plot histogram

random.seed(1)
box = [1]*72 + [0]*128

proportion = []
for _ in range(1000):
    # sample with replacement
    sample = random.choices(box, k=200)
    proportion.append(100 * sum(sample) / len(sample))

ci_interval = np.percentile(proportion, [5, 95])
print(f"Minimum proportion: {min(proportion):.2f}")
print(f"Maximum proportion: {max(proportion):.2f}")
print(f"90%-confidence interval: {ci_interval}")

fig, ax = plt.subplots(figsize=(8, 4))
pd.Series(proportion).plot.hist(bins=20, ax=ax)
ax.set_xlabel("Proportion favorable [%]")
ax.set_ylabel("Frequency")
for ci in ci_interval:
    ax.axvline(ci, color="black")
ax.plot(ci_interval, [180, 180], color="black", linestyle="--")
ax.text(ci_interval.mean(), 170, "90%-confidence interval", ha="center")
plt.show()

## Sampling distribution for a mean
# Load toyota.csv dataset and create a histogram

toyota = pd.read_csv("toyota.txt", header=None)
toyota.columns = ["price"]

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(toyota["price"], bins=6)
ax.set_xlabel("Price")
ax.set_ylabel("Frequency")
plt.show()

## Example: Toyota Corolla prices
# Load toyota.csv dataset and create a histogram

toyota = pd.read_csv("toyota.txt", header=None)
toyota.columns = ["price"]

# Single resample with replacement

random.seed(123)
sample = random.choices(toyota["price"], k=20)
print(pd.DataFrame({
    "original data": toyota["price"],
    "resample": sample,
}))
print(np.mean(sample))

# Resampling procedure

random.seed(123)

def resampleMeans(data, nResamples=1000, nSamples=20):
    means = []
    for _ in range(nResamples):
        sample = random.choices(data, k=nSamples)
        means.append(np.mean(sample))
    return means
means = resampleMeans(toyota["price"])
ci_interval = np.percentile(means, [5, 95])
print(ci_interval)

# Histogram of resampled means

fig, ax = plt.subplots(figsize=(8, 4))
pd.Series(means).plot.hist(bins=20, ax=ax)
ax.set_xlabel("Resampled mean price")
ax.set_ylabel("Frequency")
for ci in ci_interval:
    ax.axvline(ci, color="black")
ax.plot(ci_interval, [160, 160], color="black", linestyle="--")
ax.text(ci_interval.mean(), 150, "90%-confidence interval", ha="center")
plt.show()

## Normal Distribution

x = np.linspace(-3.5, 3.5, 401)
y = stats.norm.pdf(x)

fig, ax = plt.subplots(figsize=[6, 4])
ax.plot(x, y, c="black")

ci_interval = [-1.6449, 1.6449]

mask = (ci_interval[0] < x) & (x < ci_interval[1])
ax.fill_between(x[mask], y[mask], color="lightgrey")

for ci in ci_interval:
    ax.axvline(ci, color="black")
    ax.text(ci, 0.0, f"{ci:.4f}",
            verticalalignment="bottom", horizontalalignment="center")
ax.plot(ci_interval, [0.5, 0.5], color="black", linestyle="--")
ax.text(0, 0.45, "90%-confidence interval", ha="center")


ax.text(-2.5, 0.1, "Area 5%", horizontalalignment="center")
ax.text(2.5, 0.1, "Area 5%", horizontalalignment="center")
ax.text(0, 0.2, "Area 90%", horizontalalignment="center")

ax.set_xlabel("x")
ax.set_ylabel("Density")
plt.show()

# Calculate the standard deviation of the resampled means:

# estimate the standard error from resample means
print(f"standard error of mean (resample): {np.std(means):.1f}")

# estimate the standard error from the data
print(f"standard error of mean (data): {np.std(toyota['price'])/np.sqrt(20):.1f}")
