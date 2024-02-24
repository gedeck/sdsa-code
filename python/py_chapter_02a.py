## Python: Data structures and operations
# Checking the data type of a value

print(type(3.14))     # <class 'float'>
print(type("Python")) # <class 'str'>
print(type(True))     # <class 'bool'>


# This is a comment
print("Hello, World!")  # This is also a comment
print("Inside a string # this is not a comment")


pi_value = 3.14
message = "Python"
is_raining = True
_formula_1 = "H2O"


print(pi_value)     # output: 3.14


pi_value = 3.14        # value is 3.14
pi_value = 3.14159     # value is now 3.14159
two_pi = 2 * pi_value  # value of two_pi is now 6.28318


print(2 + 3)             # output: 5
print(2.0 - 3)           # output: -1.0
print(10 * 3 / 6 + 4)    # output: 9.0
print(2 ** 3)            # output: 8 (2 to the power of 3)
print(10 * 3 / (6 + 4))  # output: 3.0

# Modulus operator returns the remainder of the division
print(2 % 3)             # output: 2
print(2.0 % 3)           # output: 2.0


print("Hello, " + "World!")  # output: Hello, World!
print("Python " * 3)         # output: Python Python Python


message = "Python"
print(message[0])     # output: P
print(message[0:2])   # output: Py
print(message[2:])    # output: thon


message = "Hello, World!"
print(message.replace("World", "Python"))  # output: Hello, Python!
print(message.upper())                     # output: HELLO, WORLD!
print(message.lower())                     # output: hello, world!
print(message.split(","))                  # output: ['Hello', ' World!']


text = "4213"
print(int(text))  # output: 4213
text = "3.1415"
print(float(text))  # output: 3.1415


import math
print(math.pi)  # output: 3.141592653589793


print(f"PI = {math.pi:.4f}")  # output: 3.1416


list1 = [0, 1, 2, 3, 4, 5, 6]
list2 = []
print(len(list1))  # output: 7
print(len(list2))  # output: 0


print(list1[0]) # output: 0
print(list1[1]) # output: 1
print(list1[-1]) # output: 6
print(list1[-2]) # output: 5


print(list1[1:3]) # output: [1, 2]
print(list1[2:])  # output: [2, 3, 4, 5, 6]
print(list1[:3])  # output: [0, 1, 2]


list1[0] = 10
print(list1)        # output: [10, 1, 2, 3, 4, 5, 6]
list1.append(7)
print(list1)        # output: [10, 1, 2, 3, 4, 5, 6, 7]
list1.insert(3, 11) # insert 11 at index 3
print(list1)        # output: [10, 1, 2, 11, 3, 4, 5, 6, 7]
list1.remove(10)    # remove first occurence of 10 from the list
print(list1)        # output: [1, 2, 11, 3, 4, 5, 6, 7]
list1.pop(2)        # remove element at index 2
print(list1)        # output: [1, 2, 3, 4, 5, 6, 7]


tuple1 = (1, 2, 3, 4, 5, 6)
tuple2 = ()
print(len(tuple1))  # output: 6
print(len(tuple2))  # output: 0
print(tuple1[0])    # output: 1
print(tuple1[1:3])  # output: (2, 3)


from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
print(p.x)  # output: 1
print(p.y)  # output: 2


from typing import NamedTuple
class Point(NamedTuple):
    x: int
    y: int


set1 = {1, 2, 3, 4, 5, 6}
set2 = set()
print(len(set1))  # output: 6
print(len(set2))  # output: 0


set1.add(7)
print(set1)  # output: {1, 2, 3, 4, 5, 6, 7}
set1.remove(7)
print(set1)  # output: {1, 2, 3, 4, 5, 6}


set1 = {1, 2, 3, 4, 5, 6}
set2 = {4, 5, 6, 7, 8, 9}
print(set1.union(set2))        # output: {1, 2, 3, 4, 5, 6, 7, 8, 9}
print(set1.intersection(set2)) # output: {4, 5, 6}
print(set1.difference(set2))   # output: {1, 2, 3}


list1 = [1, 2, 3, 4, 5, 6, 1, 2, 3]
print(list1)  # output: [1, 2, 3, 4, 5, 6, 1, 2, 3]
list1 = list(set(list1))
print(list1)  # output: [1, 2, 3, 4, 5, 6]


states = {"VA": "Virginia", "MD": "Maryland", "DC": "District of Columbia"}
print(states["VA"])  # output: Virginia
print(states["MD"])  # output: Maryland


states["NY"] = "New York"
print(states)  # output: {"VA": "Virginia", "MD": "Maryland",
              #          "DC": "District of Columbia", "NY": "New York"}
del states["NY"]
print(states)  # output: {"VA": "Virginia", "MD": "Maryland",
              #          "DC": "District of Columbia"}


from collections import defaultdict
word_counts = defaultdict(int)
for word in ["apple", "banana", "apple", "banana", "apple"]:
    word_counts[word] += 1
print(word_counts)  # output: defaultdict(<class 'int'>, {'apple': 3, 'banana': 2})


class Person:
    def __init__(self, first_name, family_name, birth_date):
        self.first_name = first_name
        self.family_name = family_name
        self.birth_date = birth_date

    def full_name(self):
        return f"{self.first_name} {self.family_name}"


person = Person("John", "Doe", "1970-01-01")
print(person.first_name)  # output: John
print(person.full_name())  # output: John Doe


pi_value: float = 3.14
message: str = "Python"
is_raining: bool = True
numbers: list[int] = []


pi_value = "3.14"  # no error
