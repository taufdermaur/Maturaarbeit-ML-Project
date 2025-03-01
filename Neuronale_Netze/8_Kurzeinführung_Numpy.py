import numpy as np

a = np.zeros((1, 4))
b = np.zeros((4, 3))
c = a.dot(b)
print(c)
print(c.shape)

e = np.zeros((2, 1))
f = np.zeros((1, 3))
g = e.dot(f)
print(g.shape)

h = np.zeros((5, 4)).T # -> .T vertauscht Zeilen und Spalten der Matrix
i = np.zeros((5, 6))
j = h.dot(i)
print(j)

