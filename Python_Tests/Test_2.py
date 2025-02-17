if 89 > 23:
    print("Eightynine is greater than twentythree")

#this is a comment.

import sys
print(sys.version)

x = 5
y = 23
z = "Hello World"
print(x)
print(y)

a = str(3)
b = int(3)
c = float(3)

print(a,b,c)

print(type(x))
print("\n")
print(type(b))
print("\n")
print(type(c))
print("\n")
print(type(z))
print("\n")


j, k, l = "Sali", "Hallo", "Grüezi"

print(j)
print("\n")
print(k)
print("\n")
print(l)
print("\n")


fruits = ["apple", "banana", "cherry"]
x, y, z = fruits
print(x)
print(y)
print(z)

x = "Python"
y = "is"
z = "awesome"
print(x,y,z)

x = "Python "
y = "is "
z = "awesome"
print(x + y + z) 

#____________________________________________

x = "awesome"

def myfunc():
  print("Python is " + x)

myfunc()

print("Python is " + x)

for x in range(6):
  print(x)