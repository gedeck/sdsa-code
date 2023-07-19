
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

fig, ax = plt.subplots(figsize=(6, 4))
ax = fat_absorption[fats].plot.box()
ax.set_xlabel("Fat type")
ax.set_ylabel("Grams of fat absorbed")
ax.set_xlim(0.5, 4.5)
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
for _ in range(10000):
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
    shuffled_var = np.var(shuffled_means)
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

### ANOVA using statsmodels

stats.f_oneway(fat_absorption["Fat 1"], fat_absorption["Fat 2"], fat_absorption["Fat 3"], fat_absorption["Fat 4"])
stats.f_oneway(*[fat_absorption[fat] for fat in fats])


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

## Two-way ANOVA
# Load the joint strength data

data = pd.read_csv("jointstrength.csv")
factors = ["Antimony", "Cooling Method"]
outcome = "Joint Strength"
data.head()

# Visualize as boxplot

import seaborn as sns
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x="Cooling Method", y="Joint Strength", hue="Antimony", data=data,
            width=0.5, palette="colorblind", ax=ax)
plt.show()

# Decomposition of the antimony data into a factor diagram
# First calculate the means of the factors and their interaction

mean_treatment1 = data.groupby("Antimony")["Joint Strength"].mean()
mean_treatment2 = data.groupby("Cooling Method")["Joint Strength"].mean()
mean_interaction = data.groupby(factors).mean()

# Determine components of the factor diagram

factor_observations = np.array(data[outcome]).reshape(12, 4, order="F")
factor_grand_average = np.full((12, 4), grand_average)
# Make sure that the order of the treatment components matches the order of the observations
factor_treatment1 = mean_treatment1.to_numpy() - factor_grand_average
factor_treatment2 = np.repeat(mean_treatment2.to_numpy(), 12).reshape(12, 4) - factor_grand_average

# Determine the interaction component and finally the residual error

partial_fit = factor_grand_average + factor_treatment1 + factor_treatment2
factor_interaction = (np.repeat(data.groupby(factors).mean().to_numpy(), 3).reshape(12, 4, order="F") -
                      partial_fit)
factor_residual_error = factor_observations - partial_fit - factor_interaction

### Two-way ANOVA using statsmodels
# The following code is used to conduct a two-way ANOVA using the \mbox

import statsmodels.api as sm
from statsmodels.formula.api import ols
formula = 'Q("Joint Strength") ~ C(Antimony) * C(Q("Cooling Method"))'
ols_model = ols(formula, data=data).fit()
sm.stats.anova_lm(ols_model)

### Relationship of antimony to joint strength
# Scatterplot of the data

fig, ax = plt.subplots(figsize=(6, 4))
data.plot.scatter(x="Antimony", y="Joint Strength", ax=ax)
ax.set_xlabel("Antimony")
ax.set_ylabel("Joint Strength")
# ax.set_xlim(-0.5, 3.5)
plt.show()

# Boxplot of the data

ax = sns.boxplot(x="Antimony", y="Joint Strength", data=data, width=0.5, palette="colorblind")
plt.show()

### Regression of joint strength on antimony
# Reduce dataset to rows with antimony values less than 10

data_reduced = data[data["Antimony"] < 10]

# Fit regression model on reduced dataset

formula = 'Q("Joint Strength") ~ Antimony'
ols_model = ols(formula, data=data_reduced).fit()
ols_model.summary2()
