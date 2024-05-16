import numpy as np
import scipy
from scipy.stats import gaussian_kde
import toolbox_02450

version = toolbox_02450.__version__
print(version)

##########################################################
# EXPLAINED VARIANCE
def varianceExplained(pcs, n):
  # pcs are principal components, n is how many to calculate the explained variance for
  return ( sum([i**2 for i in pcs[0:n]]) ) / (sum([i**2 for i in pcs]))


# PROJECTION ON PRINCIPAL COMPONENTS
# v = np.array( [[-0.99, -0.13, -0.00],
#               [-0.09, 0.70, -0.71],
#               [0.09, -0.70, -0.71]])
# x = np.array([3-7/3, 2-4/3, 1-5/3]) #Mean is subtracted in this case
# print(np.dot(x, v))

def calculateCovarianceAndCorrelation(x1, x2):
  covariance = np.cov(x1, x2)[0, 1]
  correlation = np.corrcoef(x1, x2)[0, 1]
  print(f'Covariance: {covariance}, Correlation: {correlation}')

def correlationMatrix(covarianceMatrix):
  return

##########################################################
# CLASSIFICATION AND REGRESSION

# CONFUSION MATRIX
def calculateConfMatrixStats(conf_matrix):
  TP, FN, FP, TN = conf_matrix

  accuracy = (TP + TN) / sum(conf_matrix)
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  TNR = TN / (TN + FP) # True Negative Rate
  FPR = FP / (FP + TN) # False Positive Rate

  print(f'Accuracy: {"{:.3f}".format(accuracy)}, Error rate: {"{:.3f}".format(1 - accuracy)}')
  print(f'Precision :{"{:.3f}".format(precision)}, Recall :{"{:.3f}".format(recall)}')
  print(f'TNR:{"{:.3f}".format(TNR)}, FPR :{"{:.3f}".format(FPR)}')
  print()

##########################################################
# DECISION TREES

# Impurity measures

def gini(node):
  c = len(node)
  N = sum(node)
  
  return 1 - sum([(node[i]/N)**2 for i in range(c)])

def purityGainGini(root, v1, v2):
  p1 = (sum(v1) / sum(root)) * gini(v1)
  p2 = (sum(v2) / sum(root)) * gini(v2)
  return gini(root) - p1 - p2



# MAHALANOBIS DISTANCE


##########################################################
# DENSITY ESTIMATION

def gaussian_KDE(vector):
  kde = gaussian_kde(vector)
  print(f'gaussian_kde: {kde(vector)}')

def relative_density(x):
  return 1 / ((1/len(x)) * (sum(x)) )

def average_relative_density(x, neighbours):
  # neighbours should be a list of lists
  return (relative_density(x) / ((1/len(neighbours)) * sum([relative_density(i) for i in neighbours])) )