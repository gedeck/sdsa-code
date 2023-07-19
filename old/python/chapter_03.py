
# Load required packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

## Example: College admissions data

df = pd.read_csv("berkeley.csv")
subset = df[df["Major"] != "Other"][df["Major"] != " "][["Major", "Gender"]]
pd.crosstab(subset["Gender"], subset["Major"], margins=True)

## Normal distribution $p > x$

x = np.linspace(-3, 3, 401)
y = stats.norm.pdf(x)

fig, ax = plt.subplots(figsize=[6, 4])
ax.plot(x, y, c="black")

ax.fill_between(x[x>1], y[x>1], color="lightgrey")

ax.text(2, 0.15, "$P(x>1)$", horizontalalignment="center")

ax.set_xlabel("x")
ax.set_ylabel("Density")
plt.show()
