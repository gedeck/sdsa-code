
# Load required packages

import random
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt



import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

def text(ax, xcenter, ycenter, text, va="center", ha="center", **kwargs):
    ax.text(xcenter, ycenter, text, va=va, ha=ha, **kwargs)

def arrow(ax, fromPosition, toPosition, **kwargs):
    arrow = FancyArrowPatch(fromPosition, toPosition, mutation_scale=20, color='grey',
                    **kwargs)
    ax.add_patch(arrow)

def box(ax, xy, width, height, **kwargs):
    kwargs['color'] = kwargs.get('color', 'grey')
    kwargs['fill'] = kwargs.get('fill', False)
    rect = Rectangle(xy, width, height, **kwargs)
    ax.add_patch(rect)

fig, ax = plt.subplots(figsize=[7, 3])

box(ax, (-5, 10), 70, 20, color="#eee", fill=True)
arrow(ax, (68, 20), (125, 20))
box(ax, (127, 10), 70, 20, color="#eee", fill=True)
content = '\n'.join([
    "Different names for the",
    "same thing:",
    "- independent variable",
    "- input variable",
    "- exogenous variable",
    "- predictor",
    "- feature",
    "- attribute",
])
text(ax, 0, 20, content, ha="left")
content = '\n'.join([
    "Different names for the",
    "same thing:",
    "- dependent variable",
    "- output variable",
    "- endogenous variable",
    "- outcome variable",
    "- response variable",
    "- target variable",
])
text(ax, 130, 20, content, ha="left")
text(ax, 95, 22, "Supposed\ncausation")

ax.set_xlim(-10, 200)
ax.set_ylim(8, 32)
ax.set_axis_off()
plt.show()

## Example: Housing prices
# Load the data

housing = pd.read_csv("boston-housing.csv")
housing.head()

# Plot the data

fig, axes = plt.subplots(ncols=3, figsize=(6, 4))
housing["CRIM"].plot.box(ax=axes[0])
axes[0].set_ylabel("Crime rate")
housing["RM"].plot.box(ax=axes[1])
axes[1].set_ylabel("Average number of rooms")
housing["MEDV"].plot.box(ax=axes[2])
axes[2].set_ylabel("Median home value")
plt.tight_layout()
plt.show()

# Calculate correlation matrix

housing.corr()

### Multiple linear regression using statsmodels

import statsmodels.formula.api as smf
model = smf.ols("MEDV ~ CRIM + RM", data=housing).fit()
print(model.summary2())

### Multiple linear regression using interaction terms

model_i = smf.ols("MEDV ~ CRIM * RM", data=housing).fit()
print(model_i.summary2())

## Assumptions
# Random distribution

data = pd.DataFrame({
    "x": [random.uniform(0, 10) for i in range(100)],
    "y": [random.uniform(0, 10) for i in range(100)],
})
fig, ax = plt.subplots(figsize=(6, 4))
data.plot.scatter(x="x", y="y", ax=ax)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()

# Visualize predicted vs. residual for model without interaction term

housing["predicted"] = model.predict(housing)
housing["residual"] =  housing["MEDV"] - housing["predicted"]

fig, ax = plt.subplots(figsize=(6, 4))
housing.plot.scatter(x="predicted", y="residual", ax=ax)
ax.set_xlabel("Predicted value")
ax.set_ylabel("Residuals")
plt.show()
housing.to_csv("boston-housing-model.csv")

# QQ-plot for residuals

fig, ax = plt.subplots(figsize=(5, 5))
stats.probplot(housing["residual"], plot=ax)
plt.show()

## Example PEFR: resampling model coefficients
# Load the data and train regression model

pefr = pd.read_csv("pefr.txt", sep="\t")
predictors = ["exposure"]
outcome = "pefr"

model = LinearRegression()
model.fit(pefr[["exposure"]], pefr["pefr"])
print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope: {model.coef_[0]:.6f}")

# Regression via resampling: single bootstrap sample

random.seed(123)
# we resample index values with replacement
box = list(pefr.index)
intercepts = []

# resample with replacement
resample = random.choices(box, k=len(box))
# train regression model
model_r = LinearRegression()
model_r.fit(pefr.loc[resample, predictors], pefr.loc[resample, outcome])

ax = pefr.plot.scatter(x="exposure", y="pefr")
ax.plot([0, 25], model.predict([[0], [25]]), color="grey")
ax.plot([0, 25], model_r.predict([[0], [25]]), color="black", linestyle="--")

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
ax.hist(coefficients[coef], bins=20)
for ci in ci_coefficients[coef]:
    ax.axvline(ci, color="black")
ax.plot(ci_coefficients[coef], [160, 160], color="black", linestyle="--")
ax.text(ci_coefficients[coef].mean(), 150, "90%-confidence interval", ha="center")

ax.set_xlabel(coef)
ax.set_ylabel("Frequency")
plt.show()

## Interpreting Software Output

model = smf.ols("pefr ~ exposure", data=pefr).fit()
print(model.summary2())

# ANOVA analysis

import statsmodels.api as sm
print(sm.stats.anova_lm(model))

## Example: Bootstrapping the Boston Housing model
# Prepare the dataset

housing = pd.read_csv("boston-housing.csv")
outcome = "MEDV"
predictors = ["CRIM", "RM"]

# implement bootstrapping procedure

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
    for name, coef in zip(predictors, model_r.coef_):
        coefficients[name].append(coef)
intercepts = np.array(intercepts)
coefficients = {coef: np.array(values) for coef, values in coefficients.items()}

# Visualize the distributions

model = LinearRegression()
model.fit(housing[predictors], housing[outcome])

def plotDistribution(ax, values, observed, xlabel):
    ax.hist(values, bins=20)
    ax.axvline(observed, color="black")
    ax.set_xlabel(xlabel)
    ci_interval = np.percentile(values, [2.5, 97.5])
    for ci in ci_interval:
        ax.axvline(ci, color="black", linestyle="--")
    ax.text(0.1, 0.9, f"{ci_interval[0]:.3f}", transform=ax.transAxes)
    ax.text(0.8, 0.9, f"{ci_interval[1]:.3f}", transform=ax.transAxes)
    return ax

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
plotDistribution(axes[0], intercepts, model.intercept_, "Intercept")
plotDistribution(axes[1], coefficients["CRIM"], model.coef_[0], "CRIM")
plotDistribution(axes[2], coefficients["RM"], model.coef_[1], "RM")
plt.show()

# Build a model using statsmodels

model = smf.ols("MEDV ~ CRIM + RM", data=housing).fit()
print(model.summary2())

## Example: Tayko Software
# Load the data

# Training data
tayko_known = pd.read_csv("Tayko-known.csv")
tayko_known.head(10)


# Holdout data
tayko_unknown = pd.read_csv("Tayko-unknown.csv")
tayko_unknown.head(10)

# Fit a regression model to the training data

# define the formula for statsmodels
# note the Q notation to handle the whitespace in 'Web order'
formula = ("Spending ~ source_a + source_b + source_r + Freq + " +
    "last_update_days_ago + Q('Web order')")
model = smf.ols(formula, data=tayko_known).fit()
print(model.summary2())

# Fit a regression model to the training data using sklearn

predictors = ["source_a", "source_b", "source_r", "Freq",
          "last_update_days_ago", "Web order"]
outcome = "Spending"

model = LinearRegression()
model.fit(tayko_known[predictors], tayko_known[outcome])

# Apply the model to the holdout data and calculate RMSE

train_predicted = model.predict(tayko_known[predictors])
holdout_predicted = model.predict(tayko_unknown[predictors])
train_rmse = np.sqrt(np.mean((train_predicted - tayko_known[outcome]) ** 2))
holdout_rmse = np.sqrt(np.mean((holdout_predicted - tayko_unknown[outcome]) ** 2))
print(f"RMSE on training data: {train_rmse:.2f}")
print(f"RMSE on holdout data: {holdout_rmse:.2f}")

# Process of calculating RMSE demonstrated on small subset

df = pd.DataFrame({
    'actual': tayko_unknown[outcome][:10],
    'predicted': holdout_predicted[:10].round(0),
})
df['residual'] = df['actual'] - df['predicted']
df['sqr. residual'] = df['residual'] ** 2
df


print(df['sqr. residual'].sum())
print(df['sqr. residual'].mean())
print(np.sqrt(df['sqr. residual'].mean()))

# Naive rule: predict the mean

naive_predicted = tayko_known[outcome].mean()
df = pd.DataFrame({
    'actual': tayko_unknown[outcome][:10],
    'predicted': naive_predicted.round(0),
})
df['residual'] = df['actual'] - df['predicted']
df['sqr. residual'] = df['residual'] ** 2


print(df['sqr. residual'].sum())
print(df['sqr. residual'].mean())
print(np.sqrt(df['sqr. residual'].mean()))
