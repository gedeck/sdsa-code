
# Load required packages

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

## Example PEFR: resampling model coefficients
# Load the data and train regression model

pefr = pd.read_csv("pefr.txt", sep="\t")
predictors = ["exposure"]
outcome = "pefr"

model = LinearRegression()
model.fit(pefr[["exposure"]], pefr["pefr"])
print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope: {model.coef_[0]:.6f}")

# Resample the data and train regression model

random.seed(123)
# we resample index values with replacement
box = list(pefr.index)
intercepts = []
coefficients = {coef: [] for coef in predictors}
for _ in range(1000):
    # resample with replacement
    resample = random.choices(box, k=len(box))
    # train regression model
    model_r = LinearRegression()
    model_r.fit(pefr.loc[resample, predictors], pefr.loc[resample, outcome])
    intercepts.append(model_r.intercept_)
    for coef in predictors:
        coefficients[coef].append(model_r.coef_[0])
intercepts = np.array(intercepts)
coefficients = {coef: np.array(values) for coef, values in coefficients.items()}

# Calculate confidence intervals for intercept

ci_interval_intercept = np.percentile(intercepts, [5, 95])
print(f"Intercept 90% CI: {ci_interval_intercept.round(2)}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(intercepts, bins=20)
for ci in ci_interval_intercept:
    ax.axvline(ci, color="black")
ax.plot(ci_interval_intercept, [160, 160], color="black", linestyle="--")
ax.text(ci_interval_intercept.mean(), 150, "90%-confidence interval", ha="center")

ax.set_xlabel("Intercept")
ax.set_ylabel("Frequency")
plt.show()

# Calculate confidence intervals for coefficients / slope

ci_coefficients = {coef: np.percentile(values, [5, 95])
                   for coef, values in coefficients.items()}
coef = "exposure"
print(f"Intercept 90% CI: {ci_coefficients[coef].round(2)}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(intercepts, bins=20)
for ci in ci_coefficients[coef]:
    ax.axvline(ci, color="black")
ax.plot(ci_coefficients[coef], [160, 160], color="black", linestyle="--")
ax.text(ci_coefficients[coef].mean(), 150, "90%-confidence interval", ha="center")

ax.set_xlabel("Intercept")
ax.set_ylabel("Frequency")
plt.show()

## Interpreting Software Output

model = smf.ols("pefr ~ exposure", data=pefr).fit()
print(model.summary2())

# ANOVA analysis

import statsmodels.api as sm
print(sm.stats.anova_lm(model))

## Example: Cholesterol and Miles Walked
# Prepare and visualize the dataset

chol = pd.DataFrame({
    "Miles": [1.5, 0.5, 3, 2.5, 5, 3.5, 4.5, 2],
    "Chol": [193, 225, 181, 164, 140, 211, 158, 178],
})
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(chol.Miles, chol.Chol, color="k")
ax.plot([0.5, 5], [219.58 - 13.628 * 0.5, 219.58 - 13.628 * 5], color="black")

ax.set_xlabel("Walking distance in miles")
ax.set_ylabel("Cholesterol in mg/dL")
plt.show()

# Fit the model

model = smf.ols("Chol ~ Miles", data=chol).fit()
print(model.summary2())

# Calculate residuals and squared residuals

chol["predicted"] = model.predict(chol)
chol["residuals"] = model.resid
chol["squared_residuals"] = chol.residuals ** 2
print(chol)

# Calculate RMSE

print(np.sqrt(np.mean(chol.squared_residuals)))

## Example: Bootstrapping the Boston Housing model
# Prepare the dataset

housing = pd.read_csv("boston-housing.csv")
outcome = "MEDV"
predictors = ["CRIM", "RM"]

# implement bootstrappin procedure

random.seed(123)

box = list(housing.index)
intercepts = []
coefficients = {coef: [] for coef in predictors}
for _ in range(1000):
    # resample with replacement
    resample = random.choices(box, k=len(box))
    # train regression model
    model_r = LinearRegression()
    model_r.fit(housing.loc[resample, predictors], housing.loc[resample, outcome])
    intercepts.append(model_r.intercept_)
    for coef in predictors:
        coefficients[coef].append(model_r.coef_[0])
intercepts = np.array(intercepts)
coefficients = {coef: np.array(values) for coef, values in coefficients.items()}

# Visualize the distributions}

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].hist(intercepts, bins=20)
axes[0].set_title("Intercept")
axes[1].hist(coefficients["RM"], bins=20)
axes[1].set_title("RM")
axes[2].hist(coefficients["CRIM"], bins=20)
axes[2].set_title("CRIM")
plt.show()

# Build a model using statsmodels

model = smf.ols("MEDV ~ CRIM + RM", data=housing).fit()
print(model.summary2())

## Example: Tayko Software
# Load the data

tayko = pd.read_csv("tayko.csv")
tayko
