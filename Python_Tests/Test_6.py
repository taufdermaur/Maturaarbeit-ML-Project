import numpy
import matplotlib.pyplot as plt

#Create an array containing 250 random floats between 0 and 5:
x = numpy.random.uniform(0.0, 5.0, 250)
plt.hist(x, 5)
plt.show()

#Create an array with 100000 random numbers, and display them using a histogram with 100 bars:
import numpy
y = numpy.random.uniform(0.0, 5.0, 100000)
plt.hist(y, 100)
plt.show()

#A typical normal data distribution:
z = numpy.random.normal(5.0, 1.0, 100000) #We specify that the mean value is 5.0, and the standard deviation is 1.0.
plt.hist(z, 100)
plt.show()