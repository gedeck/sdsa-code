

import pandas as pd
import statsmodels.formula.api as smf

housing = pd.read_csv("boston-housing.csv")
formula = "MEDV ~ CRIM + RM"
model = smf.ols(formula, data=housing).fit()
print(model.summary2())


new_data = pd.DataFrame({"CRIM": [2, 10], "RM": [8, 7]})
predictions = model.predict(new_data)
print(predictions)


formula = "MEDV ~ CRIM + RM + CRIM:RM"
model_interaction = smf.ols(formula, data=housing).fit()
print(model_interaction.summary2())



fig, axes = plt.subplots(ncols=2, figsize=[8, 4])
df = pd.DataFrame({"Predicted value": model.fittedvalues,
                   "Actual value (MEDV)": housing["MEDV"]})
ax = df.plot.scatter(x="Actual value (MEDV)", y="Predicted value",
    alpha=0.5, ax=axes[0])
ax.set_title("(a) Main effects model")
ax.plot([0,30], [0,30], color="black")

df = pd.DataFrame({"Predicted value": model_interaction.fittedvalues,
                   "Actual value (MEDV)": housing["MEDV"]})
ax = df.plot.scatter(x="Actual value (MEDV)", y="Predicted value",
                     alpha=0.5, ax=axes[1])
ax.set_title("(b) Interactions model")
ax.plot([0,30], [0,30], color="black")
plt.show()



import scipy.stats as stats

fig, axes = plt.subplots(ncols=2, figsize=[8, 4])
ax = axes[0]
ax.scatter(model_interaction.fittedvalues, model_interaction.resid, alpha=0.5)
ax.axhline(0, color="black", linestyle="dashed")
ax.set_xlabel("Fitted values")
ax.set_ylabel("Residuals")
ax.set_title("(a) Residual plot")

ax = axes[1]
stats.probplot(model_interaction.resid, dist="norm", plot=ax)
ax.get_lines()[0].set_color("C0")
ax.get_lines()[0].set_alpha(0.5)
ax.get_lines()[1].set_color("black")
ax.set_title("(a) QQ-plot")
plt.tight_layout()
plt.show()


from sklearn.linear_model import LinearRegression
import pandas as pd

predictors = ["CRIM", "RM"]
outcome = "MEDV"
X = housing[predictors]
y = housing[outcome]
model = LinearRegression()
model.fit(X, y)
print(model.intercept_)
print(pd.Series(model.coef_, index=predictors))


new_data = pd.DataFrame({"CRIM": [2, 10], "RM": [8, 7]})
predictions = model.predict(new_data)
print(predictions)


X["CRIM:RM"] = X["CRIM"] * X["RM"]
model_interaction = LinearRegression()
model_interaction.fit(X, y)
print(model_interaction.intercept_)
print(pd.Series(model_interaction.coef_, index=predictors + ["CRIM:RM"]))


from sklearn.linear_model import LinearRegression
import numpy as np

rng = np.random.default_rng(seed=321)

housing["RANDOM"] = rng.random(len(housing))
predictors = ["CRIM", "RM", "RANDOM"]
outcome = "MEDV"
X = housing[predictors]
y = housing[outcome]

model = LinearRegression()
model.fit(X, y)
actual = pd.Series([model.intercept_, *model.coef_],
                   index=["Intercept", *predictors])

resamples = []
for _ in range(1000):
    model.fit(X, rng.permutation(y))
    resamples.append((model.intercept_, *model.coef_))
resamples = pd.DataFrame(resamples, columns=["Intercept", *predictors])



fig, axes = plt.subplots(ncols=4, figsize=[10, 3])
for ax, name in zip(axes, resamples.columns):
    resamples[name].plot.hist(bins=30, ax=ax)
    ax.axvline(actual[name], color="black", linestyle="dashed")
    ax.set_xlabel(name)
plt.tight_layout()
plt.show()


from numpy.random import RandomState
from sklearn.utils import resample

rng = np.random.RandomState(seed=321)
model = LinearRegression()
model.fit(X, y)
estimate = pd.Series([model.intercept_, *model.coef_],
                     index=["Intercept", *predictors])
coefficients = []
for _ in range(1000):
    X_resampled, y_resampled = resample(X, y, random_state=rng)
    model = LinearRegression()
    model.fit(X_resampled, y_resampled)
    coefficients.append([model.intercept_, *model.coef_])
coefficients = pd.DataFrame(coefficients, columns=estimate.index)

conf_intervals = pd.DataFrame({
    "Coefficient": estimate,
    "Lower": np.percentile(coefficients, 2.5, axis=0),
    "Upper": np.percentile(coefficients, 97.5, axis=0)
})
print(conf_intervals)


formula = "MEDV ~ CRIM + RM + RANDOM"
model = smf.ols(formula, data=housing).fit()
print(model.summary2())



fig, axes = plt.subplots(ncols=4, figsize=[10, 3])
for ax, name in zip(axes, coefficients.columns):
    coefficients[name].plot.hist(bins=30, ax=ax)
    ax.axvline(actual[name], color="black")
    ax.axvline(conf_intervals.loc[name, "Lower"], color="black", linestyle="dashed")
    ax.axvline(conf_intervals.loc[name, "Upper"], color="black", linestyle="dashed")
    ax.set_xlabel(name)
plt.tight_layout()
plt.show()
