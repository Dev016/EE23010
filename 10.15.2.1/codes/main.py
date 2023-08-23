import numpy as np
simlen = 1000
x = np.random.choice(np.arange(0,5),size=simlen)
y = np.random.choice(np.arange(0,5),size=simlen)
z = x-y
p1 = np.count_nonzero(z==0)
print("Same Day:", p1/simlen)
p2 = np.count_nonzero((z==1) | (z == -1))
print("Consecutive Days:" , p2/simlen)
print("Different Days:", (simlen - p1)/simlen)
