import numpy as np
from numpy.linalg import norm
import scipy
import toolbox_02450

version = toolbox_02450.__version__
print(version)

##########################################################
# EXPLAINED VARIANCE
def varianceExplained(pcs, n):
  # pcs are principal components, n is how many to calculate the explained variance for
  return ( sum([i**2 for i in pcs[0:n]]) ) / (sum([i**2 for i in pcs]))

pca3_2 = [17, 15.2, 13.1, 13.0, 11.8, 11.3]
# print('Question 3.2: ', varianceExplained(pca3_2, 3))

pca_3_3 = [9.7, 6.7, 5.7, 3.7, 3.0, 1.3, 0.7]
# print('Question 3.3 a: ', varianceExplained(pca_3_3, 1))
# print('Question 3.3 b: ', varianceExplained(pca_3_3, 3))
# print('Question 3.3 c: ', 1 - varianceExplained(pca_3_3, 6))
# print('Question 3.3 5: ', varianceExplained(pca_3_3, 5))

pca_3_6 = [2.69, 2.53, 1.05, 0.83, 0.49, 0.31]
# print('Question 3.6a: ', varianceExplained(pca_3_6, 1))
# print('Question 3.6b: ', varianceExplained(pca_3_6, 3))
# print('Question 3.6c: ', 1 - varianceExplained(pca_3_6, 4))

# PROJECTION ON PRINCIPAL COMPONENTS
# Question 3.5
v = np.array( [[-0.99, -0.13, -0.00],
              [-0.09, 0.70, -0.71],
              [0.09, -0.70, -0.71]])
x = np.array([3-7/3, 2-4/3, 1-5/3]) #Mean is subtracted in this case
# print(np.dot(x, v))

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

# Question 8.4
conf_LR = [12, 69, 10, 215] #[TP, FN, FP, TN]
conf_DT = [26, 65, 34, 191]
print('Logistic')
calculateConfMatrixStats(conf_LR)
print('Decision tree')
calculateConfMatrixStats(conf_DT)

# IMPURITY GINI
node = [225, 81]
def gini(node):
  c = len(node)
  N = sum(node)
  
  return 1 - sum([(node[i]/N)**2 for i in range(c)])