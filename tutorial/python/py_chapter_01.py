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
