import numpy as np
from numpy.linalg import norm
import scipy
import toolbox_02450

version = toolbox_02450.__version__
print(version)

# IMPURITY GINI
node = [225, 81]
def gini(node):
  c = len(node)
  N = sum(node)
  
  return 1 - sum([(node[i]/N)**2 for i in range(c)])

print(gini(node))