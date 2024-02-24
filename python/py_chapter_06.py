
## Counting
### Counting in Python

# create a list of 100 coin flips
import random
random.seed(1234)
coin_flips = [random.choice(["H", "T"]) for i in range(100)]

count = 0
for coin_flip in coin_flips:
    if coin_flip == "H":
        count += 1
print(count)


sum(coin_flip == "H" for coin_flip in coin_flips)


coin_flips.count("H")


counts = {"H": 0, "T": 0}
for coin_flip in coin_flips:
    counts[coin_flip] += 1
print(counts)  # prints: {'H': 55, 'T': 45}


from collections import Counter
counts = Counter(coin_flips)
print(counts)  # prints: Counter({'H': 55, 'T': 45})
print(f'Number of heads: {counts["H"]}')  # prints: Number of heads: 55


counts = Counter(coin_flips)
print(counts)  # prints: Counter({'H': 55, 'T': 45})
counts.update(random.choice(["H", "T"]) for i in range(100))
print(counts)  # prints: Counter({'H': 101, 'T': 99})

### Counting in Pandas

import pandas as pd
df = pd.read_csv("microUCBAdmissions.csv")
counts = df["Admission"].value_counts()
print(f"Number of admitted students: {counts['Admitted']}")
counts


counts = df[["Admission", "Gender"]].value_counts()
counts


print(f'Number of admitted male students: {counts["Admitted", "Male"]}')
print(counts["Admitted"])
print(counts[:, "Female"])


counts = df[["Admission", "Gender"]].value_counts(normalize=True)
counts


counts.reset_index()

### Two-way tables

df = pd.read_csv("microUCBAdmissions.csv")
pd.crosstab(df["Admission"], df["Gender"])


pd.crosstab(df["Admission"], df["Gender"], margins=True)


pd.crosstab(df["Admission"], df["Gender"], normalize="all", margins=True)


pd.crosstab(df["Admission"], df["Gender"], normalize="index", margins=True)


pd.crosstab(df["Admission"], df["Gender"], margins=True, normalize="columns")

### Chi-square test

data = pd.DataFrame({
    "states": ["Texas"] * 200 + ["California"] * 200,
    "votes": ["yes"] * 25 + ["no"] * 175 + ["yes"] * 17 + ["no"] * 183
})
observed = pd.crosstab(data["states"], data["votes"])
common_rate = observed.sum(axis=0) / 2
observed_difference = abs(observed - common_rate).sum().sum()
print(f'The observed difference is {observed_difference}')


import random
import numpy as np
random.seed(1234)
differences = []
votes = list(data["votes"])
for _ in range(5_000):
    random.shuffle(votes)
    distribution = pd.crosstab(data["states"], votes)
    differences.append(abs(distribution - common_rate).sum().sum())
at_least_observed = sum(np.array(differences) >= observed_difference) / len(differences)
print(f'Observed difference of at least {observed_difference}: {at_least_observed:.1%}')


from scipy import stats
result = stats.chi2_contingency(observed)
print(f"chi2 = {result.statistic:.3f}")
print(f"p-value = {result.pvalue:.4f}")
print(f"degrees of freedom = {result.dof}")
print("expected")
print(result.expected_freq)
