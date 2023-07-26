
## Coin toss experiment

import random
from collections import Counter

# set random seed for reproducibility (you can use any number)
random.seed(123)

coin = ["H", "T"]

# one trial: simulate 10 coin tosses and count number of heads and tails
trial = random.choices(coin, k=10)
counts = Counter(trial)
print(counts)

# keep track of numbers of heads
nr_heads = [counts["H"]]


# repeat trial 11 more times
for _ in range(11):
    trial = random.choices(coin, k=10)
    counts = Counter(trial)
    print(counts)
    nr_heads.append(counts["H"])


# repeat trial 100 more times
for _ in range(100):
    trial = random.choices(coin, k=10)
    counts = Counter(trial)
    nr_heads.append(counts["H"])
seven_or_more = sum(n >= 7 for n in nr_heads)
print(f"Seven or more heads occurred in {seven_or_more} trials")

## Proportion of heads as a function of coin tosses

import pandas as pd
nr_of_tosses = 0
nr_of_heads = 0
results = []
for increment in [10, 100, 1000]:
    for _ in range(10 if increment == 10 else 9):
        tosses = random.choices(coin, k=increment)
        nr_of_tosses += len(tosses)
        nr_of_heads += sum(toss == "H" for toss in tosses)
        results.append({"tosses": nr_of_tosses, "heads": nr_of_heads})
df = pd.DataFrame(results)
df["proportion"] = df["heads"] / df["tosses"]
df

# Visualize the results

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
df.plot(x="tosses", y="proportion", legend=False, ax=ax)
ax.set_xscale("log")
ax.set_ylim(0.25, 0.75)
ax.axhline(0.5, color="grey")
ax.axvline(100, linestyle=":")
ax.axvline(1000, linestyle=":")
ax.set_xlabel("Number of coin tosses")
ax.set_ylabel("Proportion of heads")
plt.show()

## Example: Hospital error reduction
# Load the data

data = pd.read_csv("hospitalerrors.csv")
data.head()

### Central location
# Calculate the mean

data.mean()

# Calculate the median

data.median()

### Variation
# Calculate the range

data.max() - data.min()

# Calculate percentiles

data.quantile([0.25, 0.5, 0.75])

# Calculate the interquartile range

data.quantile(0.75) - data.quantile(0.25)

# Calculate the mean absolute deviation

data.mad()

# Calculate the variance

data.var()

# Calculate the population variance; divide by $N-1$

data.var(ddof=0)

## Example: Musical genre preferences
# Create a data frame with the data

data = pd.DataFrame({
    "Rock": [7, 4, 9],
    "Hip-Hop": [1, 9, 1],
    "Country": [9, 1, 7],
    "Jazz": [1, 3, 2],
    "New Age": [3, 1, 2],
}, index=["A", "B", "C"])
data

# Calculate the Euclidean distance between A and C

import numpy as np
np.sqrt(np.sum((data.loc["A"] - data.loc["C"])**2))

# and between B and C

np.sqrt(np.sum((data.loc["B"] - data.loc["C"])**2))

# The distance between A and C is less than the distance between B and C, so A and C are more similar.
## Example: Reduction in major errors in hospitals}

data = pd.read_csv("hospitalerrors_2.csv")

# Calculate the mean reduction in errors for the treatment and control groups

mean_reduction = data.groupby("Treatment")["Reduction"].mean()
mean_reduction

# The difference between the two means is the test statistic

print(f"Difference between the two means: {mean_reduction[1] - mean_reduction[0]:.2f}")

### Frequency tables

counts = data.groupby("Treatment")["Reduction"].value_counts()
frequency = pd.DataFrame({
    "Control": counts.loc[0],
    "Treatment": counts.loc[1],
}).fillna(0)
frequency["Total"] = frequency["Control"] + frequency["Treatment"]
frequency.loc["All"] = frequency.sum()
frequency

### Cumulative frequency table

cum_frequency = pd.DataFrame({
    "Frequency": frequency.loc[1:9, "Control"],
})
cum_frequency["Cumulative Frequency"] = cum_frequency["Frequency"].cumsum()
cum_frequency["Relative Frequency"] = cum_frequency["Frequency"] / cum_frequency["Frequency"].sum()
cum_frequency

## Data Visualization
### Histogram

# determine counts data
counts = data[data["Treatment"] == 1]["Reduction"].value_counts()
# add zero values for error reductions of 7 and 8
counts.loc[7] = 0
counts.loc[8] = 0
counts = counts.sort_index()

fig, ax = plt.subplots(figsize=(6, 3))
counts.plot.bar(ax=ax)
ax.set_xlabel("Error reduction")
ax.set_ylabel("Number of hospitals")
plt.show()


fig, ax = plt.subplots(figsize=(6, 3))
data[data["Treatment"] == 1]["Reduction"].plot.hist(bins=20, ax=ax)
ax.set_xlabel("Error reduction")
ax.set_ylabel("Number of hospitals")
plt.show()

### Boxplots

fig, ax = plt.subplots(figsize=(5, 5))
data.groupby("Treatment").boxplot(subplots=False, column="Reduction", ax=ax)
ax.set_xlabel("Treatment")
ax.set_ylabel("Error reduction")
plt.show()

# a. Prepare the dataset and count the number of `H` in each run

runs = [
    'HHHTTHTTHH', 'TTHHHTTTHH', 'TTHHTHHTTT', 'HTTHTHHTTT', 'HTHTHTHTTT',
    'TTHTTTTHHH', 'HHHTHTHHHH', 'HHTHHHTTTH', 'HHTTTTHTTT', 'THHTTTHTTH',
    'TTHHHTTHHT', 'TTTHHHHTHT', 'TTHTHTTTTT', 'THTHHHTTTT', 'THTHHHTTTT',
    'HHTTHTHHHH', 'HTTHTHTHTH', 'THHTHHHHHT', 'THHHTTHTTT', 'THTHHHTHTH'
]
counts = [run.count('H') for run in runs]

# b. Prepare the frequency table

freq_table = pd.Series(counts).value_counts().sort_index()
freq_table.index.name = 'Number of H'
freq_table.name = 'Frequency'
freq_table

# c.

# solution 1
print(f'{sum(count > 7 for count in counts) / len(counts):.3f}')
# solution 2
print(f'{freq_table[7:].sum() / freq_table.sum():.3f}')

# d.
# Create several families with 10 children and determine the number of families with three or fewer girls

import random
random.seed(1234)  # set random seed for reproducibility
nrepeats = 1000
nchildren = 10

# create families with 10 children using resampling
nfamilies = 0
for _ in range(nrepeats):
    family = random.choices(['M', 'F'], k=10)  # sample with replacement
    number_of_girls = family.count('F')
    if number_of_girls <= 3:  # count the families with 3 or fewer girls
        nfamilies += 1
percentage = nfamilies / nrepeats

print(f'Percentage of families with three or fewer girls: {percentage:.2%}%')

# As an alternative, we can use the binomial distribution to calculate the probability.

from scipy.stats import binom

print(f'Percentage of families with three or fewer girls: {binom.cdf(3, 10, 0.5):.2%}%')
