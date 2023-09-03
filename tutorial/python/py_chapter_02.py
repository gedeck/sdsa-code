

x = 123
if x < 0:
    print('x is negative')
elif x == 0:
    print('x is zero')
else:
    print('x is positive')


for x in [1, 2, 3, 4, 5]:
    print(x)


x = 1
while x <= 5:
    print(x)
    x += 1


for x in [1, 2, 3, 4, 5]:
    if x == 3:
        break
    print(x)


for x in range(1, 11):
    if x % 2 == 0:
        continue
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
