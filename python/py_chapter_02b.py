

x = 123
if x < 0:
    print('x is negative')
elif x == 0:
    print('x is zero')
else:
    print('x is positive')


x = -123
if x < 0:
    x = -x
print(f'Absolute value of x: {x}')


x = 123
if x < 0:
    print('x is negative')
else:
    if x == 0:
        print('x is zero')
    else:
        print('x is positive')


for x in [1, 2, 3, 4, 5]:
    print(x)


x = 1
while x <= 5:
    print(x)
    x += 1


for x in range(1, 11):
    if x % 2 == 0:
        continue
    print(x)


for x in [1, 2, 3, 4, 5]:
    if x == 3:
        break
    print(x)


numbers = [12, 8, 9, 10, 11, 13, 9, 11, 10, 12]
sum_of_numbers = 0
for x in numbers:
    sum_of_numbers += x
mean = sum_of_numbers / len(numbers)
print(f'Mean: {mean}')


variance = 0
for x in numbers:
    variance += (x - mean) ** 2
variance /= len(numbers)
sd = variance ** 0.5
print(f'Variance: {variance}')
print(f'Standard deviation: {sd}')


greater_than_mean = []
for x in numbers:
    if x > mean:
        greater_than_mean.append(x)
print(f'Numbers greater than mean: {greater_than_mean}')


squared_differences = [(x - mean) ** 2 for x in numbers]
variance = sum(squared_differences) / len(numbers)
print(f'Variance: {variance}, standard deviation: {variance ** 0.5}')


variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)


greater_than_mean = [x for x in numbers if x > mean]


import numpy as np
numbers = list(range(1_000_000))
numbers_np = np.arange(1_000_000)


%%timeit -n 1 -r 5
sum_of_numbers = 0
for x in numbers:
    sum_of_numbers += x


%%timeit -n 1 -r 5
sum_of_numbers = sum(numbers)


%%timeit -n 10 -r 5
sum_of_numbers = numbers_np.sum()


import numpy as np
numbers = [1, 2, 3, 4, 5]
x = np.array(numbers)
print(x)


import numpy as np
x = np.array([1, 2, 3, 4, 5])
y = np.array([6, 7, 8, 9, 10])
print(x + y)
print(x * y)
print(np.sqrt((x - 3) ** 2))
print(x > 3)
print(x[x > 3])


import numpy as np
x = np.array([1, 2, 3, 4, 5])
print(x.sum())
print(x.mean())
print(x.cumsum())


import pandas as pd
df = pd.read_csv("hospitalerrors.csv")
df.head()


df.to_csv("data.csv", index=False)


# Accessing a column
control = df["Control"]
control = df.Control
# Accessing using row index and column names
control = df.loc[:, "Control"]
row = df.loc[0]  # or df.loc[0, :]
values = df.loc[4:10, "Treatment"]
value = df.loc[0, "Control"]
# Accessing data using row and column numbers
treatment = df.iloc[:, 1]
row = df.iloc[0, :]
value = df.iloc[10, 0]


# Adding or changing columns
df["Constant value"] = 1
df["Sequence"] = range(len(df))
df["NewColumn"] = df["Control"] + df["Treatment"]
df["NewColumn"] = df["NewColumn"] * 2
# Removing a column
df = df.drop(columns=["NewColumn"])
# Renaming columns
df = df.rename(columns={"Control": "ControlGroup", "Treatment": "TreatmentGroup"})
# Sorting the data
df = df.sort_values(by="ControlGroup")
