
## Python: Data visualization

import matplotlib.pyplot as plt



# create a data set
import numpy as np
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# pyplot interface
plt.plot(x, y1)
plt.title("(a) pyplot interface")
plt.show()



# object-oriented interface
fig, ax = plt.subplots()
ax.plot(x, y1)
ax.set_title("(b) object-oriented interface")
plt.show()



# create a figure
fig, ax = plt.subplots()
ax.plot(x, y1, color="C2", label="sin(x)")
ax.plot(x, y2, color="C3", label="cos(x)", linestyle=":")
ax.set_xlabel("x")
ax.set_ylabel("y = f(x)")
ax.set_title("sine and cosine")
ax.grid()
ax.legend()
plt.show()


# create a data frame
import pandas as pd
df = pd.DataFrame({"x": x, "sin(x)": y1, "cos(x)": y2})
# create a line plot
ax = df.plot(x="x", y=["sin(x)", "cos(x)"])
ax.set_title("sine and cosine")
ax.set_ylabel("y = f(x)")
plt.show()



# create a line plot
fig, axes = plt.subplots(ncols=2, figsize=(6, 3))
df.plot(x="x", y="sin(x)", ax=axes[0])
df.plot(x="x", y="cos(x)", ax=axes[1])
axes[0].set_ylabel("sin(x)")
axes[1].set_ylabel("cos(x)")
plt.tight_layout()
plt.show()



data = pd.read_csv("hospitalerrors_2.csv")
error_reduction = data[data["Treatment"] == 1]["Reduction"]
fig, ax = plt.subplots(figsize=(6, 3))
bins = [b + 1.5 for b in range(10)]
error_reduction.plot.hist(bins=bins, ax=ax, edgecolor="black")
ax.set_xlabel("Error reduction")
ax.set_ylabel("Number of hospitals")
plt.show()



baseball = pd.read_csv("baseball_payroll.csv")
baseball.plot.scatter(x="Average Payroll (Million)", y="Total Wins")
ax.set_xlabel("Average Payroll (Million)")
ax.set_ylabel("Total Wins")
plt.show()



hospital_sizes = pd.read_csv("hospitalsizes.csv")
ax = hospital_sizes["size"].plot.box()
ax.set_ylabel('Hospital size')
plt.show()



data = pd.read_csv("hospitalerrors_2.csv")
fig, ax = plt.subplots(figsize=(5, 3.5))
axes = data[["Reduction", "Treatment"]].plot.box("Treatment", ax=ax)
axes["Reduction"].set_xlabel("Treatment")
axes["Reduction"].set_ylabel("Error reduction")
plt.show()



from scipy import stats
housing = pd.read_csv("boston-housing-model.csv")
fig, ax = plt.subplots(figsize=(5, 5))
stats.probplot(housing["residual"], plot=ax)
plt.show()



import seaborn as sns

baseball = pd.read_csv("baseball_payroll.csv")
fig, ax = plt.subplots(figsize=(5, 5))
sns.regplot(x="Average Payroll (Million)", y="Total Wins", data=baseball, ax=ax)
