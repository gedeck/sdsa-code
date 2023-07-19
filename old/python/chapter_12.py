
# Load required packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Example: Baseball payroll

baseball = pd.read_csv("baseball_payroll.csv")
baseball.head()

# Visualize the dataset

fig, ax = plt.subplots(figsize=(6, 4))
baseball.plot.scatter(x="Average Payroll (Million)", y="Total Wins", ax=ax)

ax.set_xlabel("Average Payroll (Million)")
ax.set_ylabel("Total Wins")
plt.show()

# Visualize the dataset

fig, ax = plt.subplots(figsize=(6, 4))
baseball.plot.scatter(x="Average Payroll (Million)", y="Total Wins", ax=ax)

x = np.array([25, 190])
ax.plot(x, 0.4 * x + 208, color="grey", linewidth=2)

ax.set_xlabel("Average Payroll (Million)")
ax.set_ylabel("Total Wins")
plt.show()

## Example: delta-wire linear regression
# Load the data

delta_wire = pd.DataFrame({
    "training": [0, 100, 250, 375, 525, 750, 875, 1100, 1300, 1450, 1660, 1900, 2300, 2600,
                2850, 3150, 3500, 4000],
    "productivity": [70_000, 70_350, 70_500, 72_600, 74_000, 76_500, 77_000, 77_400, 77_900,
                    77_200, 78_900, 81_000, 82_500, 84_000, 86_500, 87_000, 88_600, 90_000],
})

# Example: Delta Wire - Linear regression using scikit-learn

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(delta_wire[["training"]], delta_wire["productivity"])
print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope: {model.coef_[0]:.6f}")

# Visualize the regression line

fig, ax = plt.subplots(figsize=(6, 4))
delta_wire.plot.scatter(x="training", y="productivity", ax=ax)
x = np.array([0, 4000])
ax.plot(x, model.predict(x.reshape(-1, 1)), color="grey", linewidth=2)
ax.set_xlabel("Training (hours)")
ax.set_ylabel("Productivity (pounds per week)")
plt.show()

# Example: PEFR - Linear regression using scikit-learn

pefr = pd.read_csv("pefr.txt", sep="\t")
model = LinearRegression()
model.fit(pefr[["exposure"]], pefr["pefr"])
print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope: {model.coef_[0]:.6f}")

# Visualize the regression line

fig, ax = plt.subplots(figsize=(6, 4))
pefr.plot.scatter(x="exposure", y="pefr", ax=ax)
ax.set_xlabel("Exposure")
ax.set_ylabel("PEFR")
x = np.array([0, 23])
ax.plot(x, model.predict(x.reshape(-1, 1)), color="grey", linewidth=2)
plt.show()

## Residual plots - Baseball payroll dataset
# Calculate the residuals

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(baseball[["Average Payroll (Million)"]], baseball["Total Wins"])
print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope: {model.coef_[0]:.6f}")

baseball["residuals"] = baseball["Total Wins"] - model.predict(baseball[["Average Payroll (Million)"]])

# Visualize the residuals

fig, ax = plt.subplots(figsize=(6, 4))
baseball.plot.scatter(x="Average Payroll (Million)", y="residuals", ax=ax)
ax.axhline(0, color="black")
ax.set_xlabel("Average Payroll (Million)")
ax.set_ylabel("Residuals")
plt.show()

## Residual plots - Delta Wire dataset
# Calculate the residuals

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(delta_wire[["training"]], delta_wire["productivity"])
delta_wire["residuals"] = delta_wire["productivity"] - model.predict(delta_wire[["training"]])

# Visualize the residuals

fig, ax = plt.subplots(figsize=(6, 4))
delta_wire.plot.scatter(x="training", y="residuals", ax=ax)
ax.axhline(0, color="black")
ax.set_xlabel("Training (hours)")
ax.set_ylabel("Residuals")
plt.show()

## Residual plots - PEFR dataset
# Calculate the residuals

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(pefr[["exposure"]], pefr["pefr"])
pefr["residuals"] = pefr["pefr"] - model.predict(pefr[["exposure"]])

# Visualize the residuals

fig, ax = plt.subplots(figsize=(6, 4))
pefr.plot.scatter(x="exposure", y="residuals", ax=ax)
ax.axhline(0, color="black")
ax.set_xlabel("Exposure")
ax.set_ylabel("Residuals")
plt.show()
