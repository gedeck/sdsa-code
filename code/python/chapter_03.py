
# Load required packages

import matplotlib.pyplot as plt
import pandas as pd

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

# Calculate the mean absolute deviation (pandas used to have a function to calculate this, but it was removed)

abs(data - data.mean()).mean()

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

## Frequency Tables

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
cum_frequency["Cumulative Relative Frequency"] = cum_frequency["Relative Frequency"] / cum_frequency["Relative Frequency"].sum()
cum_frequency

## Data Visualization
### Histogram

# determine counts data
counts = data[data["Treatment"] == 1]["Reduction"].value_counts()
# add zero values for error reductions of 7 and 8
counts.loc[7] = 0
counts.loc[8] = 0
counts = counts.sort_index()



# we want bins to be centered on integers and therefore need to specify the bin edges
bins = [b + 1.5 for b in range(10)]
fig, ax = plt.subplots(figsize=(6, 3))
data[data["Treatment"] == 1]["Reduction"].plot.hist(bins=bins, ax=ax, edgecolor="black")
ax.set_xlabel("Error reduction")
ax.set_ylabel("Number of hospitals")
plt.show()

### Boxplots

axes = data[['Reduction', 'Treatment']].plot.box('Treatment')
axes[0].set_xlabel("Treatment")
axes[0].set_ylabel("Error reduction")
plt.show()

## Histograms
### Histograms for lists of values

hospital_sizes = pd.read_csv('hospitalsizes.csv')
bins = [i * 100 for i  in range(12)]
ax = hospital_sizes['size'].plot.hist(bottom=0, bins=bins, edgecolor='black')
ax.set_xlim(0, 1150)
plt.show()

### Histograms for frequency tables
# If you already have a frequency table, you can use the `DataFrame.plot.bar` method. While this method is normally used to create a bar chart with categories on the $x$ axis, we can change its appearance to look like a histogram by setting the bar width to 1. In order to get a correct axis, you need to make sure that all bins have values in the frequency table. Here, this applies to the bar at a hospital size of 950.

hospital_sizes_ft = pd.read_csv('hospitalsizes_frequencytable.csv')
ax = hospital_sizes_ft.plot.bar(x='size', y='frequency', width=1, edgecolor="black", legend=False)
ax.set_xlabel("Hospital size")
ax.set_ylabel("Number of hospitals")
plt.show()

## Box Plots

hospital_sizes = pd.read_csv('hospitalsizes.csv')
ax = hospital_sizes['size'].plot.box()
ax.set_ylabel('Hospital size')
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

# Create a random population of 1000 values from a normal distribution with mean 100 and standard deviation 15

import random
random.seed(123)
population = [random.gauss(100, 15) for _ in range(1000)]

# Calculate the standard deviation of the population

# calculate the standard deviation of the population
pop_std = np.std(population, ddof=0)
print(f'Population standard deviation: {pop_std:.2f}')

# Calculate the standard deviation of resamples of size 10 using $N$ and $N-1$  in the Denominator

resample_std_N = []
resample_std_N_1 = []
for _ in range(1000):
    resample = random.sample(population, 10)
    resample_std_N.append(np.std(resample, ddof=0))
    resample_std_N_1.append(np.std(resample, ddof=1))
    mean_resample_std_N = np.mean(resample_std_N)
    mean_resample_std_N_1 = np.mean(resample_std_N_1)
    print(f'Mean sample standard deviation (N): {mean_resample_std_N:.2f}')
    print(f'Mean sample standard deviation (N-1): {mean_resample_std_N_1:.2f}')

# Plot the distribution of these standard deviations for both cases and compare to the standard deviation of the original population

import seaborn as sns
fig, ax = plt.subplots(figsize=(6, 3))
sns.histplot(resample_std_N, color='C0', ax=ax, label='N', alpha=0.2,
             edgecolor='lightgrey')
sns.histplot(resample_std_N_1, color='C1', ax=ax, label='N-1', alpha=0.2,
             edgecolor='lightgrey')
ax.axvline(pop_std, color='k', label='Population', linewidth=3)
ax.axvline(mean_resample_std_N, color='C0', label='Resample N', linewidth=3)
ax.axvline(mean_resample_std_N_1, color='C1', label='Resample N-1', linewidth=3)
ax.set_xlabel('Standard deviation')
ax.set_ylabel('Density')
ax.legend()
plt.show()

# We can see that the standard deviation of the resamples using $N-1$ in the denominator is closer to the population standard deviation.
