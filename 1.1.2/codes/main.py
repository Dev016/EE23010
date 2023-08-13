import numpy as np
import math
B = np.array([-4,6])
C = np.array([-3,-5])

V1 = B - C
V2 = V1.reshape(-1,1)
print("The length of BC is:")
print(math.sqrt(V1@V2))
