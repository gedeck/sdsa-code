
# Load required packages

import matplotlib.pyplot as plt
import numpy as np



from matplotlib.patches import Circle, Rectangle
fig, ax = plt.subplots(figsize=(4, 2))
ax.add_patch(Rectangle((0, 0), width=4, height=2, color='lightgrey'))
ax.add_patch(Circle((1, 1), radius=0.75, color='C1', fill='C2', lw=2))
ax.text(1, 1, 'E', ha='center', va='center', fontsize=20)
ax.text(3, 1, '~E', ha='center', va='center', fontsize=20)
ax.set_xlim(0, 4)
ax.set_ylim(0, 2)
ax.axis('off')



from matplotlib.patches import Rectangle
w, h = 4, 2
x1, x2, x3, x4 = 0.2, 1.5, 2.2, 3.5
y1, y2, y3, y4 = 0.2, 0.5, 1.3, 1.6
fig, ax = plt.subplots(figsize=(4, 3))
ax.add_patch(Rectangle((0, 0), width=w, height=h, color='lightgrey'))
ax.add_patch(Rectangle((x1, y1), width=(x3-x1), height=(y3-y1), color='C1'))
ax.text(0.5*(x1+x2), 0.5*(y1+y3), 'E', ha='center', va='center', fontsize=20)
ax.add_patch(Rectangle((x2, y2), width=(x4-x2), height=(y4-y2), color='C2'))
ax.text(0.5*(x3+x4), 0.5*(y2+y4), 'B', ha='center', va='center', fontsize=20)
ax.add_patch(Rectangle((x2, y2), width=(x3-x2), height=(y3-y2), color='C0'))
ax.text(0.5*(x2+x3), 0.5*(y2+y3), 'B$\cap$E', ha='center', va='center', fontsize=13)
ax.set_xlim(0, w)
ax.set_ylim(0, h)
ax.axis('off')
plt.show()



import random
import pandas as pd

random.seed(123)
nrepeat = 10_000
nr_heads = []
for _ in range(nrepeat):
    nr_head = random.choices(["H", "T"], k=150).count("H")
    nr_heads.append(nr_head)

ax = pd.Series(nr_heads).plot.hist(bins=np.arange(49.5, 100, 1))
ax.axvline(75, color='black', linestyle='--')
ax.set_xlabel("Number of heads")

## The normal distribution

from scipy import stats
x = np.linspace(-4.5, 4.5, 1000)
y = stats.norm.pdf(x, 0, 1)
fig, ax = plt.subplots(figsize=(8, 4))
ax.axhline(0, color="grey")
ax.axvline(0, color="grey")
ax.plot(x, y, linewidth=2)
ax.set_xlabel("Standard deviations")
ax.set_ylabel("")
ax.set_xlim(-4.5, 4.5)
plt.show()

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
