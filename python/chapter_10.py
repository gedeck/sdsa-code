
# Load required packages

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def f(x):
    return 2.5 * x + 1

fig, ax = plt.subplots(figsize=(4, 3))
x = np.array([0, 6])
y = f(x)
ax.plot(x, y, color="black")
ax.plot([2, 3, 3], [f(2), f(2), f(3)])
ax.text(2.5, 5, "$\Delta x$")
ax.text(3.2, 0.5*(f(2)+f(3)), "$\Delta y$")
ax.text(3.75, 6, r"Slope: $a=\frac{\Delta y}{\Delta x}$")
ax.set_xlim(0, 5.5)
ax.set_ylim(0, 12)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.annotate("Intercept: b", xy=(0, 1), xytext=(0.75, 1), va="center",
            arrowprops={"facecolor": "black", "width": 1, "headwidth": 5})

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



from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    "x": [0.5, 2, 3, 5, 6],
    "y": [1, 0, 5, 2, 7],
})
model = LinearRegression()
model.fit(df[["x"]].to_numpy(), df["y"].to_numpy())
ax = df.plot.scatter(x="x", y="y", c="C0")
ax.plot([0, 7], model.predict([[0], [7]]), color="black")
for i, row in df.iterrows():
    ax.plot([row["x"], row["x"]],
            [row["y"], model.predict([[row["x"]]])[0]], color="grey")
ax.text(1.9, 4, "Residual")
ax.text(2.75, 2, "Regression line")
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
predictor = ["Average Payroll (Million)"]
model = LinearRegression()
model.fit(baseball[predictor], baseball["Total Wins"])
print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope: {model.coef_[0]:.6f}")

baseball["predicted"] = model.predict(baseball[predictor])
baseball["residuals"] = baseball["Total Wins"] - baseball["predicted"]

# Visualize the residuals

fig, ax = plt.subplots(figsize=(6, 4))
baseball.plot.scatter(x="predicted", y="residuals", ax=ax)
ax.axhline(0, color="black")
ax.set_xlabel("Predicted Total Wins")
ax.set_ylabel("Residuals")
plt.show()

## Residual plots - Delta Wire dataset
# Calculate the residuals

from sklearn.linear_model import LinearRegression
predictors = ["training"]
model = LinearRegression()
model.fit(delta_wire[predictors], delta_wire["productivity"])
delta_wire["predicted"] = model.predict(delta_wire[predictors])
delta_wire["residuals"] = delta_wire["productivity"] - delta_wire["predicted"]

# Visualize the residuals

fig, ax = plt.subplots(figsize=(6, 4))
delta_wire.plot.scatter(x="predicted", y="residuals", ax=ax)
ax.axhline(0, color="black")
ax.set_xlabel("Predicted Productivity (hours)")
ax.set_ylabel("Residuals")
plt.show()

## Residual plots - PEFR dataset
# Calculate the residuals

from sklearn.linear_model import LinearRegression
predictors = ["exposure"]
model = LinearRegression()
model.fit(pefr[predictors], pefr["pefr"])
pefr["predicted"] = model.predict(pefr[predictors])
pefr["residuals"] = pefr["pefr"] - pefr["predicted"]

# Visualize the residuals

fig, ax = plt.subplots(figsize=(6, 4))
pefr.plot.scatter(x="predicted", y="residuals", ax=ax)
ax.axhline(0, color="black")
ax.set_xlabel("Predicted pefr")
ax.set_ylabel("Residuals")
plt.show()



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

fig, ax = plt.subplots(figsize=[6, 3])

box(ax, (-5, 10), 50, 20, color="#eee", fill=True)
text(ax, 20, 27, "The data", fontsize=16)
text(ax, 20, 20, r"$x_1, x_2, \dots, x_n, y$")

text(ax, 100, 20, r"$y = b_0 + b_1x_1+b_2x_2 + \dots$")

text(ax, 180, 27, "Assess", fontsize=16)
text(ax, 180, 20, "$p$-values\nConf. interval\n$R^2$\n$F$\nRMSE")
arrow(ax, (46, 20), (65, 20))
arrow(ax, (136, 20), (155, 20))

ax.set_xlim(-10, 200)
ax.set_ylim(8, 32)
ax.set_axis_off()
plt.tight_layout()



fig, ax = plt.subplots(figsize=[7.5, 4])

box(ax, (-5, 15), 50, 15, color="#eee", fill=True)
text(ax, 20, 27, "The data", fontsize=16)
text(ax, 20, 20, r"$x_1, x_2, \dots, x_n, y$")

box(ax, (75, 25), 50, 10, color="#eee", fill=True)
text(ax, 100, 30, "Data to fit\nmodel")

box(ax, (75, 10), 50, 10, color="#eee", fill=True)
text(ax, 100, 15, "Data to assess\nmodel")

text(ax, 50, 22.5, "(1) Randomly\nsplit data", ha="left")
arrow(ax, (46, 25), (74, 30))
arrow(ax, (46, 20), (74, 15))

text(ax, 150, 30, "(2) Fit a model\n$y = b_0 + b_1x_1+b_2x_2 + \dots$", ha="left")
arrow(ax, (126, 30), (149, 30))
text(ax, 150, 15, "(3) Apply model", ha="left")
arrow(ax, (126, 15), (149, 15))
text(ax, 232, 15, "(4) Compare\npredicted values to\nactual values,\ncalculate RMSE", ha="left")
arrow(ax, (205, 15), (230, 15))

ax.set_xlim(-10, 310)
ax.set_ylim(8, 40)
ax.set_axis_off()
plt.tight_layout()
