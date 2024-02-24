
## Implement simple random sample (SRS)

import random
import matplotlib.pyplot as plt

random.seed(123)
box = [0] * 128 + [1] * 72
# Repeated sampling
nsamples = 1000
proportion = []
for _ in range(nsamples):
    nr_ones = 0
    for _ in range(len(box)):
        random.shuffle(box)
        sample = random.choice(box)
        if sample == 1:
            nr_ones += 1
    proportion.append(nr_ones / len(box) * 100)
# Visualize the results
fig, ax = plt.subplots()
ax.hist(proportion, bins=20)
ax.set_xlabel("Proportion favorable [\%]")
ax.set_ylabel("Frequency")
plt.show()
print(f"Mean: {sum(proportion)/len(proportion)}")

# Improved version of the sampling experiment

proportion = []
for _ in range(nsamples):
    samples = random.choices(box, k=len(box))
    nr_ones = sum(samples)
    proportion.append(nr_ones / len(box))
# Visualize the results
fig, ax = plt.subplots()
ax.hist(proportion, bins=20)
plt.show()
print(f"Mean: {sum(proportion)/len(proportion)}")

# Using weights instead of a box

import numpy as np
population_size = 200
proportion = []
for _ in range(nsamples):
    samples = random.choices([0, 1], weights=[128, 72], k=population_size)
    proportion.append(sum(samples) / population_size)
print(f"Mean: {np.mean(proportion)}")

### Determining confidence intervals

percentiles = [0.025, 0.975]
sorted_proportion = sorted(proportion)
lower = sorted_proportion[round(percentiles[0] * nsamples)]
upper = sorted_proportion[round(percentiles[1] * nsamples)]
print(f"Confidence interval: [{lower:.3f}, {upper:.3f}]")

# Using `np.percentile` to determine the confidence interval

import numpy as np
lower, upper = np.percentile(proportion, [2.5, 97.5])
print(f"Confidence interval: [{lower:.3f}, {upper:.3f}]")

# Using the normal approximation to determine the confidence interval

from scipy import stats
p = np.mean(proportion)
n = population_size
z = stats.norm().ppf(0.975)
lower = p - z * np.sqrt(p * (1 - p) / n)
upper = p + z * np.sqrt(p * (1 - p) / n)
print(f"Confidence interval: [{lower:.3f}, {upper:.3f}]")


from statsmodels.stats.proportion import proportion_confint
ci_interval = proportion_confint(72, 200, alpha=0.05, method="normal")
print(f"Confidence interval (normal): [{ci_interval[0]:.3f}, {ci_interval[1]:.3f}]")
ci_interval = proportion_confint(72, 200, alpha=0.05, method="beta")
print(f"Confidence interval (beta): [{ci_interval[0]:.3f}, {ci_interval[1]:.3f}]")

## Bootstrap sampling to determine confidence intervals for a mean

import pandas as pd
rng = np.random.default_rng(seed=321)

df = pd.read_csv("toyota.txt", header=None, names=["price"])

nsamples = 1000
mean_price = []
for _ in range(nsamples):
    sample = df.sample(frac=1.0, replace=True, random_state=rng)
    mean_price.append(sample["price"].mean())

percentiles = [0.025, 0.975]
lower, upper = np.percentile(mean_price, [2.5, 97.5])
print(f"Confidence interval: [{lower:.2f}, {upper:.2f}]")



# Visualize the results
fig, ax = plt.subplots()
ax.hist(mean_price, bins=20)
ax.set_xlabel("Mean price")
ax.set_ylabel("Frequency")
ax.axvline(lower, color="black", linestyle="--")
ax.axvline(upper, color="black", linestyle="--")
ax.axvline(np.mean(mean_price), color="black")
plt.show()


from scipy import stats

mean_price = np.mean(df["price"])
std_err_mean = stats.sem(df["price"])
dof = len(df["price"]) - 1
# calculate 95% confidence interval
ci_interval = stats.t.interval(0.95, dof, loc=mean_price, scale=std_err_mean)

print(f'95% confidence interval: [{ci_interval[0]:.2f}, {ci_interval[1]:.2f}]')

## Stratified sampling

rng = np.random.default_rng(seed=123)
df = pd.read_csv("microUCBAdmissions.csv")
resample = (df.groupby("Major")
    .sample(frac=0.1, random_state=rng))
print(f"{len(resample)} rows sampled from {len(df)} rows")
resample.head()


pd.DataFrame({
    "Original": df.value_counts("Major") / len(df),
    "Resampled": resample.value_counts("Major") / len(resample)
}).round(3)

## Stratified bootstrap sampling for multiple categorical variables

rng = np.random.default_rng(seed=123)
resample = (df.groupby(["Major", "Gender"])
    .sample(frac=1, replace=True, random_state=rng)
    .reset_index(drop=True))
print(f"{len(resample)} rows sampled from {len(df)} rows")
resample.head()


print(pd.DataFrame({
    "Original": df.value_counts(["Gender", "Major"]),
    "Resampled": resample.value_counts(["Gender", "Major"]),
}).sort_index())


rng = np.random.default_rng(seed=123)
df = pd.read_csv("boston-housing.csv")

resample = (df.groupby(pd.cut(df["CRIM"], bins=10),
                       observed=True)
    .sample(frac=0.1, random_state=rng))
print(f"{len(resample)} rows sampled from {len(df)} rows")
resample.head()



bw_method = 0.5
ax = df["CRIM"].plot.density(color="black", bw_method=bw_method,
                             label="Full dataset")
resample["CRIM"].plot.density(linestyle="--", ax=ax, bw_method=bw_method,
                              label="Stratified sample")
df["CRIM"].sample(frac=0.1, random_state=rng).plot.density(linestyle=":",
                              ax=ax, bw_method=bw_method,
                              label="Random sample")
