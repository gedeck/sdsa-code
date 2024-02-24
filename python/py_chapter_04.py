
## Python: Random numbers

# set the seed to ensure a different outcome
import random
random.seed()

### Generating random numbers using the `random` package
# generate a random number between 0 and 1

import random
print(random.random(), random.random())

# set a random seed to ensure that the same sequence is generated each time

random.seed(123)
print(random.random(), random.random())
random.seed(123)
print(random.random(), random.random())


print(random.randrange(0, 10, 2)) # 6 - output from 2, 4, 6, or 8
print(random.randint(0, 10))      # 4 - output from 0, 1, ..., 9, or 10
print(random.choice([0, 1, 2]))   # 0 - output from 0, 1, or 2


x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
random.shuffle(x)
print(x) # [7, 8, 3, 4, 1, 9, 2, 5, 6, 0]
print(random.sample(x, k=3)) # [4, 3, 7]
print(random.choices(x, k=6)) # [0, 5, 6, 4, 7, 5]

### Generating random numbers using `numpy` and `scipy`
# Initalize a random number generator

import numpy as np
rng = np.random.default_rng(seed=321)
print(rng.random()) # 0.6587666953866232
print(rng.random()) # 0.9083083579615744


print(rng.integers(low=0, high=10, size=3)) # [6 4 3]
print(rng.uniform(low=0, high=10, size=3)) # [5.55856507 9.00575242 6.82980572]


# sample with replacement
print(rng.choice([0, 1, 2, 3, 4], size=3)) # [3 3 0]
# sample without replacement
print(rng.choice([0, 1, 2, 3, 4], size=3, replace=False)) # [2 1 0]


from scipy.stats import norm
rng = np.random.default_rng(seed=123) # create a RNG with a fixed seed for reproducibility
print(norm.rvs(loc=0.0, scale=1.0, size=3, random_state=rng))
# [-0.98912135 -0.36778665  1.28792526]
print(norm.pdf(x=0.0, loc=0.0, scale=1.0))
# 0.3989422804014327
print(norm.cdf(x=0.0, loc=0.0, scale=1.0))
# 0.5

### Using random numbers in other packages

import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [6, 7, 8, 9, 10]})
print(df.sample(n=3, random_state=123))


import pandas as pd
df = pd.read_csv("hospitalerrors_2.csv")
print(df.head())


mean_reduction = df[['Treatment', 'Reduction']].groupby("Treatment").mean()
print(mean_reduction)


observed_difference = (mean_reduction.loc[1, 'Reduction'] -
                       mean_reduction.loc[0, 'Reduction'])
print(f'Observed reduction {observed_difference:.3f}')


observation = df['Reduction']
treatment = df['Treatment']


import random
random.seed(123)
shuffled = list(observation)
random.shuffle(shuffled)

# split the shuffled observations by treatment group
observed_0 = []
observed_1 = []
for obs, treat in zip(shuffled, treatment):
    if treat == 0:
        observed_0.append(obs)
    else:
        observed_1.append(obs)

# calculate the mean reduction for the treatment and control group
obs_treatment_0 = sum(observed_0) / len(observed_0)
obs_treatment_1 = sum(observed_1) / len(observed_1)

# calculate the difference
obs_difference = obs_treatment_1 - obs_treatment_0
print(f'Observed difference after shuffling: {obs_difference:.3f}')


observed_0 = [obs for obs, treat in zip(shuffled, treatment) if treat == 0]
observed_1 = [obs for obs, treat in zip(shuffled, treatment) if treat == 1]


import numpy as np
obs_treatment_0 = np.mean(observed_0)
obs_treatment_1 = np.mean(observed_1)


shuffled = observation.copy()
random.shuffle(shuffled)
means = shuffled.groupby(treatment).mean()
means[1] - means[0]


shuffled = observation.copy() # create a copy of the observations
differences = []
for _ in range(1000):
    random.shuffle(shuffled)  # shuffle the copy
    means = shuffled.groupby(treatment).mean()
    differences.append(means[1] - means[0])

print(f"Mean difference after reshuffling {np.mean(differences):.2f}")
print(f"Minimum difference {np.min(differences):.2f}")
print(f"Maximum difference {np.max(differences):.2f}")



ax = pd.Series(differences).plot.hist(bins=25)
ax.axvline(x=observed_difference, color='grey')
plt.show()


nr_greater_observed = sum(d >= observed_difference for d in differences)
prob_observed =  nr_greater_observed / len(differences)
print("Probability of observing a difference of 0.92 or larger by chance: "
      f"{prob_observed:.1%}")

### Write functions for code reuse

def resampling_difference_means(observations, treatments, nr_trials=1000):
    """ Calculate differences in means between two treatment groups using resampling """
    # create an independent copy of the observations
    shuffled = pd.Series(observations)
    differences = []
    for _ in range(nr_trials):
        random.shuffle(shuffled)  # shuffle the copy
        means = shuffled.groupby(treatments).mean()
        differences.append(means[1] - means[0])
    return differences


differences = resampling_difference_means(df['Reduction'], df['Treatment'], nr_trials=2000)
print(f"Mean difference after reshuffling {np.mean(differences):.2f}")
print(f"Minimum difference {np.min(differences):.2f}")
print(f"Maximum difference {np.max(differences):.2f}")


import random
import pandas as pd

def resampling_difference_means(observations, treatments, nr_trials=1000):
    """ Calculate differences in means between two treatment groups using resampling """
    # calculate observed difference
    observed = observations.groupby(treatments).mean()
    # create an independent copy of the observations
    shuffled = pd.Series(observations)
    differences = []
    for _ in range(nr_trials):
        random.shuffle(shuffled)  # shuffle the copy
        means = shuffled.groupby(treatments).mean()
        differences.append(means[1] - means[0])
    return {"observed": observed[1] - observed[0],
            "resamples": differences}


import numpy as np
import pandas as pd
import resampling

df = pd.read_csv("hybrid.csv")
differences = resampling.resampling_difference_means(df["Measurement"], df["Experiment"])

resamples = differences["resamples"]
print(f"Observed difference {differences['observed']:.2f}")
print(f"Mean difference after reshuffling {np.mean(resamples):.2f}")
print(f"Minimum difference {np.min(resamples):.2f}")
print(f"Maximum difference {np.max(resamples):.2f}")


from resampling import resampling_difference_means


from resampling import *
