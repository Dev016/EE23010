import numpy as np
simlen = 100000
x = np.random.choice(np.arange(0,5),size=simlen)
y = np.random.choice(np.arange(0,5),size=simlen)
z = x-y
p1 = np.count_nonzero(z==0)
print("Probability through Simulation")
print("Same Day:", p1/simlen)
p2 = np.count_nonzero((z==1) | (z == -1))
print("Consecutive Days:" , p2/simlen)
print("Different Days:", (simlen - p1)/simlen)

print("Theoretical Probability")
p1 = np.array([0.2,0.2,0.2,0.2,0.2])@(np.array([0.2,0.2,0.2,0.2,0.2]))
print("Same Day:" , p1)
p2 = np.array([0.2,0.2,0.2,0.2,0])@(np.array([0.2,0.2,0.2,0.2,0.2])) + np.array([0,0.2,0.2,0.2,0.2])@(np.array([0.2,0.2,0.2,0.2,0.2]))
print("Consecutive Days:" , p2)
print("Different Days:", 1- p1)
