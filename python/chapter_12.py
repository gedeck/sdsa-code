
# Load required packages

import matplotlib.pyplot as plt

## Riding lawnmower example
# Load the data.

import pandas as pd
mower_df = pd.read_csv("RidingMowers.csv")

# Define new customer at (60, 20)

import pandas as pd
new_customer = pd.DataFrame({"Income": 60, "Lot_Size": 20},
                            index=["New customer"])
new_customer

# Visualize dataset with new customer at (60, 20)

import matplotlib.pyplot as plt

owners = mower_df[mower_df["Ownership"] == "Owner"]
non_owners = mower_df[mower_df["Ownership"] == "Nonowner"]

def basic_mower_plot():
    fig, ax = plt.subplots(figsize=(6, 4))
    owners.plot.scatter(x="Income", y="Lot_Size", c="C0", s=30, ax=ax)
    ax.plot(non_owners["Income"], non_owners["Lot_Size"], color="black", marker="o",
            fillstyle="none", linestyle="none")
    new_customer.plot.scatter(x="Income", y="Lot_Size", c="C2", marker="x", s=50, ax=ax)
    return ax
basic_mower_plot()

# Calculate distances of new customer to dataset

import numpy as np
predictors = ["Income", "Lot_Size"]
distances = mower_df[predictors] - new_customer[predictors].to_numpy()
mower_df["distance"]= np.sqrt(((distances/mower_df[predictors].std())**2).sum(axis=1))
neighbors_by_distance = mower_df.sort_values(by="distance")

# Identify closest neighbor $k=1$

ax = basic_mower_plot()
new_c = new_customer.to_numpy().flatten()
for k in range(1):
    neighbor = neighbors_by_distance[predictors].iloc[k, :].to_numpy()
    ax.plot([new_c[0], neighbor[0]], [new_c[1], neighbor[1]], color="grey", zorder=0)
    ax.scatter(neighbor[0], neighbor[1], color="lightgrey", s=200, zorder=0)

# Identify closest neighbors $k=5$

ax = basic_mower_plot()
new_c = new_customer.to_numpy().flatten()
for k in range(5):
    neighbor = neighbors_by_distance[predictors].iloc[k, :].to_numpy()
    ax.plot([new_c[0], neighbor[0]], [new_c[1], neighbor[1]], color="grey", zorder=0)
    ax.scatter(neighbor[0], neighbor[1], color="lightgrey", s=200, zorder=0)

## Hypothetical example
# Create the data frame

import pandas as pd
df = pd.DataFrame([
    [1, 1, 1, 1, 0, 1, 0],
    [1, 1, 1, 1, 0, 1, 0],
    [1, 1, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1, 0, 1],
    [0, 0, 1, 0, 1, 0, 1],
],
index=["1", "2", "3", "4", "5", "6"],
columns=["zinc10", "zinc90", "mag10", "mag90",
         "cotton10", "cotton90", "Registry"])

# Calculate Euclidean distance to new customer

import numpy as np
predictors = [ "zinc10", "zinc90", "mag10", "mag90",
        "cotton10", "cotton90"]
df[predictors]

new_customer = np.array([1, 0, 1, 1, 0, 1])
distances = np.sqrt(np.sum((df[predictors] - new_customer) ** 2, axis=1))
min_distances = np.where(distances == distances.min())

closest_customers = df.iloc[min_distances[0], :]
closest_customers
