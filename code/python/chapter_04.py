
# Load required packages

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

## Permutation Test

data = pd.read_csv("hospitalerrors_2.csv")
data.head()

# Calculate difference of means for a permuation of the data

# set seed for reproducibility
random.seed(123)

data["shuffled"] = random.sample(list(data["Reduction"]), k=len(data))
randomized_means = data.groupby("Treatment")["shuffled"].mean()
difference = randomized_means[1] - randomized_means[0]
print(f"Difference after reshuffling {difference:.2f}")

# Repeat reshuffling 1000 times.

# set seed for reproducibility
random.seed(123)

# define a function for the resampling experiment
def resample_experiment(observation, treatment, repeats):
    differences = []
    observation = list(observation)
    for _ in range(repeats):
        # sample from the list without replacement to shuffle the data
        shuffled = pd.Series(random.sample(observation, k=len(observation)))
        randomized_means = shuffled.groupby(treatment).mean()
        difference = randomized_means[1] - randomized_means[0]
        differences.append(difference)
    return pd.Series(differences)

differences = resample_experiment(data["Reduction"], data["Treatment"], 1000)
print(f"Mean difference after reshuffling {np.mean(differences):.2f}")
print(f"Minimum difference {np.min(differences):.2f}")
print(f"Maximum difference {np.max(differences):.2f}")

# keep the result of the resampling experiment
resamples = {"repeat 1": differences}

# Plot the distribution of differences

fig, ax = plt.subplots(figsize=(8, 4))
pd.Series(differences).plot.hist(bins=25, ax=ax)
ax.set_xlabel("Difference in means")
ax.set_ylabel("Frequency")
ax.axvline(x=0.92, color="grey")
plt.show()

# Calculate the probability of getting a difference of 0.92 or more

print("  ".join([f"{v:.2f}" for v in sorted(differences, reverse=True)[:25]]))
# we round here to compensate for small differences in floating point numbers
sum(differences.round(2) >= 0.92), sum(differences.round(2) >= 0.92) / len(differences)

# Repeat permuation experiment with a different random seed

random.seed(456)
differences = resample_experiment(data["Reduction"], data["Treatment"], 1000)
resamples["repeat 2"] = differences
print("  ".join([f"{v:.2f}" for v in sorted(differences, reverse=True)[:25]]))
sum(differences.round(2) >= 0.92), sum(differences.round(2) >= 0.92) / len(differences)

# Repeat three more times and display the results

resamples["repeat 3"] = resample_experiment(data["Reduction"], data["Treatment"], 1000)
resamples["repeat 4"] = resample_experiment(data["Reduction"], data["Treatment"], 1000)
resamples["repeat 5"] = resample_experiment(data["Reduction"], data["Treatment"], 1000)

import seaborn as sns

df = pd.DataFrame(resamples)
ax = sns.boxplot(data=df, palette="colorblind", notch=True, width=0.5)
ax.axhline(0.92, color="grey")
plt.show()

# Calculate the probability of getting a difference of 0.92 or more

np.sum(np.sum(df.round(2) >= 0.92)) / (1000 * 5)

### Increase the number of repeats to estimate the p-value

random.seed(123)
differences = resample_experiment(data["Reduction"], data["Treatment"], 20_000)
above_threshold = np.sum(differences.round(2) >= 0.92)
p_value = above_threshold / len(differences)
print(f"p-value: {p_value:.4f}")
print(f"Number of permuations greater or equal 0.92: {above_threshold}")

## The normal distribution

from scipy import stats
x = np.linspace(-4.5, 4.5, 1000)
y = stats.norm.pdf(x, 0, 1)
fig, ax = plt.subplots(figsize=(8, 4))
ax.axhline(0, color="grey")
ax.axvline(0, color="grey")
ax.plot(x, y, linewidth=2)
ax.set_xlabel("Standard deviations")
ax.set_ylabel("")
ax.set_xlim(-4.5, 4.5)
plt.show()
