import numpy as np
import matplotlib.pyplot as plt
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
p0 = np.array([0.2,0.2,0.2,0.2,0.2])@(np.array([0.2,0.2,0.2,0.2,0.2]))
print("Same Day:" , p0)
p1 = np.array([0.2,0.2,0.2,0.2,0])@(np.array([0.2,0.2,0.2,0.2,0.2])) + np.array([0,0.2,0.2,0.2,0.2])@(np.array([0.2,0.2,0.2,0.2,0.2]))
p2 = np.array([0.2,0.2,0.2,0,0])@(np.array([0.2,0.2,0.2,0.2,0.2])) + np.array([0,0,0.2,0.2,0.2])@(np.array([0.2,0.2,0.2,0.2,0.2]))
p3 = np.array([0.2,0.2,0,0,0])@(np.array([0.2,0.2,0.2,0.2,0.2])) + np.array([0,0,0,0.2,0.2])@(np.array([0.2,0.2,0.2,0.2,0.2]))
p4 = np.array([0.2,0,0,0,0])@(np.array([0.2,0.2,0.2,0.2,0.2])) + np.array([0,0,0,0,0.2])@(np.array([0.2,0.2,0.2,0.2,0.2]))
print("Consecutive Days:" , p1)
print("Different Days:", 1- p1)

#plt.bar(x = [0,1,2,3,4],height = [p0,p1,p2,p3,p4])
#plt.show()

plt.stem([0,1,2,3,4], [p0,p1,p2,p3,p4], markerfmt='o', linefmt='C1-', use_line_collection=True)
plt.xlabel('$|Z|$')
plt.ylabel('$Probability$')
plt.title("Probability Distribution")
plt.grid()
plt.savefig('/home/devansh/EE23010/10.15.2.1/figs/figure1.png')

