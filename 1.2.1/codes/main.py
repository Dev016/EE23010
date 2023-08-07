import numpy as np

#The following points are given
A = np.array([1,-1])
B = np.array([-4,6])
C = np.array([-3,-5])

#We need to find the mid points D, E,F of the sides BC, CA and AB respectively.
# D = (B+C)/2

D = (B + C)/2

#Similarly for E and F
E = (A + C)/2
F = (A + B)/2

print("D:", list(D))
print("E:", list(E))
print("F:", list(F))

