
# Load required packages

import matplotlib.pyplot as plt

## Coin toss experiment

import random
from collections import Counter

# set random seed for reproducibility (you can use any number)
random.seed(123)

coin = ["H", "T"]

# one trial: simulate 10 coin tosses and count number of heads and tails
trial = random.choices(coin, k=10)
counts = Counter(trial)
print(counts)

# keep track of numbers of heads
nr_heads = [counts["H"]]


# repeat trial 11 more times
for _ in range(11):
    trial = random.choices(coin, k=10)
    counts = Counter(trial)
    print(counts)
    nr_heads.append(counts["H"])


# repeat trial 100 more times
for _ in range(100):
    trial = random.choices(coin, k=10)
    counts = Counter(trial)
    nr_heads.append(counts["H"])
seven_or_more = sum(n >= 7 for n in nr_heads)
print(f"Seven or more heads occurred in {seven_or_more} trials")

# Create the visualization for the results of three coin tosses

import matplotlib.pyplot as plt
import itertools

# create all possible combinations of coin toss results (1, 1, 1), (1, 1, -1), ...
flip_results = list(itertools.product([-1, 1], repeat=3))

# y-positions at which coins are to be drawn
shifts = [4, 2, 1]
fig, ax = plt.subplots(figsize=[5, 3])
props = {'facecolor': '#dddddd', 'edgecolor': 'grey', 'boxstyle': 'circle'}
centered_circle = {"verticalalignment": "center", "horizontalalignment": "center",
                "bbox": {"facecolor": "#dddddd", "edgecolor": "grey", "boxstyle": "circle"}}
for flip_result in flip_results:
    # determine y positions for the coins in the tree and draw lines
    deltas = [0]
    for direction, shift in zip(flip_result, shifts):
        deltas.append(deltas[-1] + direction * shift)
    ax.plot([0, 1, 2, 3], deltas, color='grey')
    # add the coins along the tree lines
    faces = ['H' if c == 1 else 'T' for c in flip_result]
    for x, y, face in zip([1, 2, 3], deltas[1:], faces):
        ax.text(x, y, face, **centered_circle)
    # add the resulting coin combination on the right
    for x, y, face in zip([3.8, 4.15, 4.5], [deltas[-1]]*3, faces):
        ax.text(x, y, face, **centered_circle)
ax.set_xlim(-0.5, 4.5)
plt.axis("off")  # hide the axis
plt.show()

## Proportion of heads as a function of coin tosses

import pandas as pd
nr_of_tosses = 0
nr_of_heads = 0
results = []
for increment in [10, 100, 1000]:
    for _ in range(10 if increment == 10 else 9):
        tosses = random.choices(coin, k=increment)
        nr_of_tosses += len(tosses)
        nr_of_heads += sum(toss == "H" for toss in tosses)
        results.append({"tosses": nr_of_tosses, "heads": nr_of_heads})
df = pd.DataFrame(results)
df["proportion"] = df["heads"] / df["tosses"]
df

# Visualize the results

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
df.plot(x="tosses", y="proportion", legend=False, ax=ax)
ax.set_xscale("log")
ax.set_ylim(0.25, 0.75)
ax.axhline(0.5, color="grey")
ax.axvline(100, linestyle=":")
ax.axvline(1000, linestyle=":")
ax.set_xlabel("Number of coin tosses")
ax.set_ylabel("Proportion of heads")
plt.show()
