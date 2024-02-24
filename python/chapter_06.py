
# Load required packages

import matplotlib.pyplot as plt
import pandas as pd

## Example: College admissions data
# Load and preprocess the data

df = pd.read_csv("microUCBAdmissions.csv")

# Create a 2 x 2 table

pd.crosstab(df["Admission"], df["Gender"], margins=True)

# Percent by column

(pd.crosstab(df["Admission"], df["Gender"], margins=True, normalize="columns") * 100).round(2)

## Example: College admissions data (continued)
# Load, process the data, and create a 2x2 table

df = pd.read_csv("microUCBAdmissions.csv")
admission_gender = pd.crosstab(df["Gender"], df["Major"], margins=True)
admission_gender

# Percentage of department applications by gender

100 * pd.crosstab(df["Gender"], df["Major"], margins=True, normalize="columns").round(4)

# Male/female applications by department

100 * pd.crosstab(df["Gender"], df["Major"], margins=True, normalize="index").round(4)



import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow

x1 = 20
x2 = 90
y = 97
fig, ax = plt.subplots(figsize=(6, 6))
ax.add_patch(Rectangle((0, 0), 100, 100, ec="black", color="white"))
ax.add_patch(Rectangle((0, 0), x1, y, ec="black", color="gold"))
ax.add_patch(Rectangle((0, y), x2, 100-y, ec="black", color="C3"))
ax.add_patch(Rectangle((x2, y), 100-x2, 100-y, ec="black", color="white"))

ax.text(0.5*(100+x1), 0.5*y, "96,903", ha="center", va="center")
ax.text(0.5*(x1), 0.5*y, "2997", ha="center", va="center")

ax.text(0.5*(x2), 110, "98", ha="center", va="center")
ax.text(0.5*(100+x2), 110, "2", ha="center", va="center")
ax.add_patch(Arrow(0.5*(x2), 107, 0, -5, color="grey"))
ax.add_patch(Arrow(0.5*(100+x2), 107, 0, -5, color="grey"))

ax.text(-20, 0.5*(100+y), "Disease\n(100)", ha="center", va="center")
ax.add_patch(Arrow(-12, 0.5*(100+y), 11, 0, color="grey"))
ax.text(-15, 0.5*(y), "No disease\n(99,900)", ha="center", va="center")
ax.plot([-2, -3, -3, -2], [0, 0, y, y], color='grey')
ax.plot([-4, -3], [y/2, y/2], color='grey')

ax.text(50, 117, "Test result", ha="center", va="center", fontsize=16)
ax.text(-40, 50, "Actual\ncondition", ha="center", va="center", fontsize=16)

ax.set_xlim(-10, 105)
ax.set_ylim(-5, 110)
ax.set_axis_off()

## Independence
# Admission rates by department

100 * pd.crosstab(df["Admission"], df["Major"], margins=True, normalize="columns").round(4)

### The chi-square distribution

import numpy as np
from scipy import stats
x = np.linspace(0, 40, 200)
distribution = pd.DataFrame({
    'x': x,
    'chi_2': stats.chi2(df=10).pdf(x),
})
ax = distribution.plot(x='x', y='chi_2', legend=False, color='black')
ax.set_xlabel('$\chi^2$ statistic')
ax.set_ylabel('probability density')
plt.show()

## Simpson's Paradox
# Berkeley admission rates

100 * pd.crosstab(df["Admission"], df["Gender"], margins=True, normalize="columns").round(4)

# Berkeley admission rates by department

admitted = df[df["Admission"] == "Admitted"]
100 * (pd.crosstab(admitted["Gender"], admitted["Major"]) /
pd.crosstab(df["Gender"], df["Major"])).round(4)
