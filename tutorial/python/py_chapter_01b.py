## Python: Data structures and operations
# Checking the data type of a value


print(type(3.14))     # <class 'float'>

print(type("Python")) # <class 'str'>

print(type(True))     # <class 'bool'>




pi_value = 3.14

message = "Python"

is_raining = True




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

