import numpy 

speed1 = [86,87,88,86,87,85,86]
x = numpy.std(speed1)
print("Standard deviation is a number that describes how spread out the values are: ",x)


speed2 = [32,111,138,28,59,77,97]
y = numpy.std(speed2)
print("Standard deviation is a number that describes how spread out the values are: ",y)

z = numpy.var(speed2)
print("Variance: ",z)