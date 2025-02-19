import numpy
from scipy import stats

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

speed.sort()
print("Sortierte Liste: ",speed)

print("\n") #Zeilenumbruch

calculate_mean = (99+86+87+88+111+86+103+87+94+78+77+85+86) / 13
print("Manuel berechneter Durchschnittswert: ",calculate_mean)

print("\n")

x = numpy.mean(speed)
print("Durchschnittswert: ",x)

print("\n")

y = numpy.median(speed)
print("Mittelpunkt: ",y)

print("\n")

z = stats.mode(speed)
print("Meistgebrauchter Wert: ",z)