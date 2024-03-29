
# Load required packages

import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

## Example: Marriage therapy

observed = pd.DataFrame({
    "B": {1: 15, 2: 3, 3: 11},
    "I": {1: 24, 2: 5, 3: 1},
}).transpose()

# use scipy.stats.chi2_contingency to determine expected values
result = stats.chi2_contingency(observed)
print(result.expected_freq)
# calculate sum of absolute differences
np.abs(observed - result.expected_freq).sum().sum()

# Resampling procedure

from collections import Counter
random.seed(123)
box = [1] * 39 + [2] * 8 + [3] * 12
statistics = []
for _ in range(10_000):
    random.shuffle(box)
    resample_B = box[:29]
    resample_I = box[29:]
    counts_B = Counter(resample_B)
    counts_I = Counter(resample_I)
    table = pd.DataFrame({
        "B": {i: counts_B[i] for i in [1, 2, 3]},
        "I": {i: counts_I[i] for i in [1, 2, 3]},
    }).transpose()
    statistic = np.abs(table - result.expected_freq).sum().sum()
    statistics.append(statistic)

statistics = np.array(sorted(statistics, reverse=True))
count_above = sum(statistics > 20.42)
p_value = count_above / len(statistics)

print(f"observed: {np.abs(observed - result.expected_freq).sum().sum():.2f}")
print(f"count above: {count_above}")
print(f"p-value: {p_value:.4f}")


random.seed(123)
box = [1] * 39 + [2] * 8 + [3] * 12
random.shuffle(box)
resample_B = box[:29]
resample_I = box[29:]
counts_B = Counter(resample_B)
counts_I = Counter(resample_I)
table = pd.DataFrame({
    "B": {i: counts_B[i] for i in [1, 2, 3]},
    "I": {i: counts_I[i] for i in [1, 2, 3]},
}).transpose()
print("Shuffled box")
print(" & ".join([str(i) for i in box]))
print(table)
print(result.expected_freq)
print(np.abs(table - result.expected_freq))
print(np.abs(table - result.expected_freq).sum().sum())

# Show a values in the vicinity of the observed value

at_threshold = np.where(statistics <= 20.42)[0][0]
for i in range(-5, 5):
    print(f"{at_threshold + i + 1} : {statistics[at_threshold + i]:.3f}")

## Chi-Square Example

result = stats.chi2_contingency(observed)
print(f"chi2 = {result.statistic:.3f}")
print(f"p-value = {result.pvalue:.4f}")
print(f"degrees of freedom = {result.dof}")
print("expected")
print(result.expected_freq)

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

print(f"Number of resamples with sum of absolute deviations >= 216:{above_216}")
print(f"p-value = {p_value:.4f}")

## ANOVA of fat absorption data
# Load required packages

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Prepare dataset

fat_absorption = pd.DataFrame([
    [1, 164, 178, 175, 155],
    [2, 172, 191, 193, 166],
    [3, 168,  197,  178,  149],
    [4,  177,  182,  171,  164],
    [5,  156,  185,  163,  170],
    [6,  195,  177,  176, 168],
], columns=["Replication", "Fat 1", "Fat 2", "Fat 3", "Fat 4"])

### Exploratory data analysis
# Dotplot

fats = ["Fat 1", "Fat 2", "Fat 3", "Fat 4"]

# convert from wide to long format for plotting
fat_long = fat_absorption.melt(value_vars=fats)

fig, ax = plt.subplots(figsize=(6, 4))
fat_long.plot.scatter(x="variable", y="value", ax=ax)
ax.set_xlabel("Fat type")
ax.set_ylabel("Grams of fat absorbed")
ax.set_xlim(-0.5, 3.5)
plt.show()

# Boxplots

ax = fat_absorption[fats].plot.box()
ax.set_xlabel("Fat type")
ax.set_ylabel("Grams of fat absorbed")
ax.set_xlim(0.5, 4.5)
plt.show()


fat_average = fat_absorption[fats].mean(axis=0)
grand_average = fat_absorption[fats].to_numpy().mean()
deviation = fat_average - grand_average
print(deviation)
deviation_squared = deviation ** 2
print(deviation_squared)
total_deviation = deviation_squared.sum()
print(total_deviation)
variance = total_deviation / (len(fats) - 1)
print(variance)

### Resampling procedure to compare means

random.seed(123)

# step 1
data = fat_absorption[fats].to_numpy()
box = list(data.flatten())

shuffled_vars = []
for _ in range(1_000):
    # step 2
    # shuffle the box and convert to same format as original
    # dataset
    shuffled = random.sample(box, k=len(box))
    shuffled_data = np.reshape(shuffled, data.shape)

    # step 3
    # determine means of shuffled data
    shuffled_means = shuffled_data.mean(axis=0)

    # step 4
    # calculate variance of means
    shuffled_var = np.var(shuffled_means, ddof=1)
    shuffled_vars.append(shuffled_var)
shuffled_above_threshold = sum(shuffled_vars > variance)
resampled_p_value = shuffled_above_threshold / len(shuffled_vars)

for resampled_var in sorted(shuffled_vars, reverse=True)[:10]:
    print(f"{resampled_var:.3f}")
print()
print(f"Resampled p-value: {resampled_p_value:.4f}")

# Visualize the resmapled variances in a histogram

ax = pd.Series(shuffled_vars).plot.hist(bins=30)
ax.set_xlabel("Variance")
ax.axvline(variance, color="black")
plt.show()

### Components of variance
# Calculate averages for each fat and grand average

fat_average = fat_absorption[fats].mean(axis=0)
grand_average = fat_absorption[fats].to_numpy().mean()

print("Average for fats")
print(fat_average)
print(f"Grand average: {grand_average}")

### Constructing the Factor Diagram

factor_observations = fat_absorption[fats]
factor_grand_average = factor_observations.copy()
factor_grand_average[:] = grand_average
factor_treatment_effects = factor_observations.copy()
factor_treatment_effects[:] = fat_average - grand_average

factor_residual_error = factor_observations - factor_grand_average - factor_treatment_effects
factor_observations, factor_grand_average, factor_treatment_effects, factor_residual_error


ssq_grand_average = (factor_grand_average**2).sum().sum()
ssq_treatment_effects = (factor_treatment_effects**2).sum().sum()
ssq_residual_error = (factor_residual_error**2).sum().sum()
print(ssq_grand_average)
print(ssq_treatment_effects)
print(ssq_residual_error)


df_treatment = len(fats) - 1
df_residual = len(fats) * (len(fat_absorption) - 1)
var_treatment_effects = ssq_treatment_effects / df_treatment
var_residual_error = ssq_residual_error / df_residual
F_statistic = var_treatment_effects / var_residual_error
p_value = 1 - stats.f.cdf(F_statistic, df_treatment, df_residual)

print(f"Variance treatment {var_treatment_effects}")
print(f"Variance residual {var_residual_error}")
print(f"F-statistic {F_statistic:.6f}")
print(f"p-value {p_value:.4f}")

### ANOVA using statsmodels

import statsmodels.api as sm
from statsmodels.formula.api import ols
ols_model = ols("value ~ C(variable)", data=fat_long).fit()
sm.stats.anova_lm(ols_model)

### ANOVA using scipy

stats.f_oneway(fat_absorption["Fat 1"], fat_absorption["Fat 2"], fat_absorption["Fat 3"], fat_absorption["Fat 4"])
stats.f_oneway(*[fat_absorption[fat] for fat in fats])

## F-distribution

from scipy import stats
x = 5.406
print(f"p-value {1 - stats.f.cdf(x, 3, 20):.4f}")
print(f"p-value {stats.f.sf(x, 3, 20):.4f}")



x = np.linspace(0, 5, 100)
df = pd.DataFrame({
    "x": x,
    "f-pdf": stats.f.pdf(x, 3, 20),
    "1-cdf": 1 - stats.f.cdf(x, 3, 20),
})
ax = df.plot(x="x", y="f-pdf", legend=False)
ax.axvline(x=F_statistic, color="grey")
ax.set_ylabel("Density")
ax.set_ylim(0, 1)
ax.set_xlim(0, 6)
ax.set_title("Probability density of $F(3, 20)$")



ax = df.plot(x="x", y="1-cdf", legend=False)
ax.axvline(x=F_statistic, color="grey")
ax.axhline(y=p_value, color="grey")
ax.set_ylabel("Density")
ax.set_ylim(0, 1)
ax.set_xlim(0, 6)
ax.set_title("Inverse cumulative distribution density of $F(3, 20)$")
