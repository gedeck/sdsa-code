
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

# Calculate the population variance; divide by $n$

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

# We can also plot the frequency table using a bar chart.

# we want bins to be centered on integers and therefore need to specify the bin edges
fig, ax = plt.subplots(figsize=(6, 3))
counts.plot.bar(edgecolor="black", width=1, ax=ax)
ax.set_xlabel("Error reduction")
ax.set_ylabel("Number of hospitals")
plt.show()

### Boxplots

axes = data[["Reduction", "Treatment"]].plot.box("Treatment")
axes["Reduction"].set_xlabel("Treatment")
axes["Reduction"].set_ylabel("Error reduction")
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

## Bar charts

df = pd.read_csv("microUCBAdmissions.csv")
admission_gender = pd.crosstab(df["Gender"], df["Major"], margins=True)
ax = admission_gender.loc["All","A":"F"].plot.bar()
ax.set_xlabel("Department")
ax.set_ylabel("Applicants")
plt.show()

# Sort the bars by number of applicants

admission_gender = admission_gender.sort_values('All', axis=1, ascending=False)
ax = admission_gender.iloc[-1, 1:].plot.bar()
ax.set_xlabel("Department")
ax.set_ylabel("Applicants")
plt.show()

## Box Plots

hospital_sizes = pd.read_csv('hospitalsizes.csv')
ax = hospital_sizes['size'].plot.box()
ax.set_ylabel('Hospital size')
plt.show()


q = np.quantile(hospital_sizes["size"], q=[0.25, 0.5, 0.75])
iqr = q[2] - q[0]
print(f"25-75: [{q[0]} {q[2]}]")
print(f"IQR: {iqr}")
print(f"Median: {q[1]}")
sizes = hospital_sizes["size"]
whisker_min = sizes[sizes > q[0] - 1.5*iqr].min()
whisker_max = sizes[sizes < q[2] + 1.5*iqr].max()
print(f"Whiskers: {whisker_min}  {whisker_max}")
