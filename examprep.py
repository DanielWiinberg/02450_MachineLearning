import numpy as np
from numpy.linalg import norm
import scipy
import toolbox_02450

version = toolbox_02450.__version__
print(version)

# x = np.array([1, 2])
# A = np.array([4, 2])

# norm_l1 = norm(x-A, 1)
# print(norm_l1)

# n12 = 8 + 15 + 5
# n21 = 7 + 11 + 17
# p = 2*scipy.stats.binom.cdf(min([n12,n21]), n=n12+n21, p=0.5)
# print(p)

x35 = np.array([-1.24, -0.26, -1.04])
x53 = np.array([-0.6, -0.86, -0.5])
print(norm(x35-x53, 4))