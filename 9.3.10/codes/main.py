import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm

n = 4  
p = 0.90  
k = np.arange(0, n+1)  
binomial_pmf = binom.pmf(k, n, p)
print("Binomial")
print(binomial_pmf[n])
mu = n * p
sigma = np.sqrt(n * p * (1 - p))
x = np.linspace(0, n, 1000)
normal_pdf = norm.pdf(x, mu, sigma)
print("Gaussian")
print(normal_pdf[len(x)-1])
plt.stem(k, binomial_pmf, label='Binomial PMF', basefmt='b-')
plt.plot(x, normal_pdf, label='Normal PDF', color='r')
plt.axvline(x=9, color='g', linestyle='--', label='x=9')
plt.xlabel('Number of Balls Drawn which are not marked zero')
plt.ylabel('Probability')
plt.legend()
plt.savefig("./figure.png")
plt.show()
