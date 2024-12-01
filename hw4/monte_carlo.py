import numpy as np

size = 10000
sample = np.cos(np.random.randn(size))**2
print(f"Expected value:\t{sample.mean():.3}\nVariance:\t{sample.var():.3}")

# 99.9% confidence level is 3.291
print(f"The 99.9% confidence interval is {sample.mean():.3f}±{3.291*sample.std()/np.sqrt(size):.3f}")