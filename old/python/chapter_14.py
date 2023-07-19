
# Load required packages

import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

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
housing["residual"] =  housing["predicted"] - housing["MEDV"]

fig, ax = plt.subplots(figsize=(6, 4))
housing.plot.scatter(x="predicted", y="residual", ax=ax)
ax.set_xlabel("Predicted value")
ax.set_ylabel("Residuals")
plt.show()

# QQ-plot for residuals

fig, ax = plt.subplots(figsize=(5, 5))
stats.probplot(housing["residual"], plot=ax)
plt.show()
