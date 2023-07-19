
# Load required packages

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

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

## Appendix A: explore changes in initial guess

random.seed(1)
def get_ci_interval(ratio_positive):
    box = [1]*int(200*ratio_positive) + [0]*int(200*(1-ratio_positive))
    proportion = []
    for _ in range(2000):
        sample = random.choices(box, k=200)
        proportion.append(100 * sum(sample) / len(sample))
    ci_interval = np.percentile(proportion, [5, 95])
    return ci_interval
percent_positives = np.linspace(30, 40, 21)
ci_intervals = [get_ci_interval(percent_positive / 100)
                for percent_positive in percent_positives]


df = pd.DataFrame(ci_intervals, columns=["lower", "upper"],
index=percent_positives)
fig, ax = plt.subplots(figsize=(8, 4))
ax.axhline(np.mean(df["lower"] - percent_positives), color="lightgrey")
ax.axhline(np.mean(df["upper"] - percent_positives), color="lightgrey")
for percent, ci_interval in df.iterrows():
    ax.plot([percent, percent], ci_interval - percent, color="black")
ax.scatter(percent_positives, [0] * len(percent_positives), color="black")
ax.set_xlabel("Percent positive")
ax.set_ylabel("Interval around percent positive")
plt.show()

## Appendix B: Parametric Bootstrap
# Example of a parametric resample

rng = np.random.default_rng(seed=123)
sample = toyota["price"]
df = pd.DataFrame({
    "original data": toyota["price"],
    "resample": stats.norm.rvs(loc=np.mean(sample), scale=np.std(sample), size=len(sample),
                               random_state=rng),
})
print(df.round(1))
print(df.mean(axis=0))

# Define function that uses parametric bootstrap to create samples

def get_resampled_means_parametric(sample, num_resamples=1000, seed=None):
    rng = np.random.default_rng(seed=seed)
    norm = stats.norm(loc=np.mean(sample), scale=np.std(sample))
    resampled_means = []
    for _ in range(num_resamples):
        resample = norm.rvs(size=len(sample), random_state=rng)
        resampled_means.append(np.mean(resample))
    return resampled_means
means = get_resampled_means_parametric(toyota["price"], num_resamples=1000, seed=123)
ci_interval = np.percentile(means, [5, 95])
print("mean: ", np.mean(means))
print("90% confidence interval: ", ci_interval)


fig, ax = plt.subplots(figsize=(8, 4))
pd.Series(means).plot.hist(bins=20, ax=ax)
ax.set_xlabel("Resampled mean price")
ax.set_ylabel("Frequency")
for ci in ci_interval:
    ax.axvline(ci, color="black")
ax.plot(ci_interval, [190, 190], color="black", linestyle="--")
ax.text(ci_interval.mean(), 175, "90%-confidence interval", ha="center")
plt.show()
