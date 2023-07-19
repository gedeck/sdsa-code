
# Load required packages

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
