import numpy as np
import scipy
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
import toolbox_02450
from scipy.stats import chi2
from scipy.stats import binom

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
  # root, v1, v2 are three arrays that specify how many is part of each class. 
  # N is the sum of the array. So root_N = v1_N + v2_N
  p1 = (sum(v1) / sum(root)) * gini(v1)
  p2 = (sum(v2) / sum(root)) * gini(v2)
  return gini(root) - p1 - p2


##########################################################
# DENSITY ESTIMATION

def gaussian_KDE(vector, bandwidth):
  kde = gaussian_kde(vector, bw_method=bandwidth)
  print(f'gaussian_kde: {kde(vector)}')

def relative_density(x):
  return 1 / ((1/len(x)) * (sum(x)) )

def average_relative_density(x, neighbours):
  # neighbours should be a list of lists
  return (relative_density(x) / ((1/len(neighbours)) * sum([relative_density(i) for i in neighbours])) )

def adaboost(weights, correctly_classified):
  """
  Calculate the updated weights for Adaboost.

  Parameters:
  weights (list or np.array): Array of weights before boosting.
  correctly_classified (list or np.array): Array containing 1 for correct classification and 0 for incorrect classification.
  """
  # Convert input lists to numpy arrays for easier manipulation
  weights = np.array(weights)
  correctly_classified = np.array(correctly_classified)
  
  # Calculate the error rate
  error_rate = np.sum(weights * (1 - correctly_classified)) / np.sum(weights)
  
  # Compute the classifier's weight
  alpha = 0.5 * np.log((1 - error_rate) / (error_rate + 1e-10))
  
  # Update the weights
  updated_weights = weights * np.exp(-alpha * correctly_classified + alpha * (1 - correctly_classified))
  
  # Normalize the weights
  updated_weights /= np.sum(updated_weights)
  
  return f'updated weights: {updated_weights}, classifier importance: {alpha}'

def detect_anomalies_kde(X_train, X_test, bandwidth=1.0, threshold=0.01):
  """
  Detect anomalies in the test vector X_test using Kernel Density Estimation 
  fitted on the training vector X_train.

  Parameters:
  X_train (list or array-like): Training vector of observations.
  X_test (list or array-like): Test vector of observations.
  bandwidth (float): The bandwidth parameter for KDE, controlling the smoothness of the density estimate.
  threshold (float): The density threshold below which an observation is considered an anomaly.

  Returns:
  list: A list of booleans where True indicates an anomaly.
  """
  # Reshape the data to fit the KDE model
  X_train = np.array(X_train).reshape(-1, 1)
  X_test = np.array(X_test).reshape(-1, 1)
  
  # Fit the KDE model on the training data
  kde = KernelDensity(bandwidth=bandwidth)
  kde.fit(X_train)
  
  # Compute the log density scores for the test data
  log_density_scores = kde.score_samples(X_test)
  
  # Convert log density scores to density scores
  density_scores = np.exp(log_density_scores)
  
  # Determine anomalies based on the threshold
  anomalies = density_scores < threshold
  
  return anomalies.tolist()

def standard_deviation(x):
  return np.sqrt(sum([(xi - np.mean(x))**2 for xi in x ]) / (len(x) - 1))

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def mcNemar(M):
  n11, n12 = M[0]
  n21, n22 = M[1]

  # Calculate the p-value using the binomial distribution
  m = min(n12, n21); n = n12 + n21
  p_value = 2 * binom.cdf(m, n, 0.5)

  # Calculate the overall accuracy
  total = n11 + n12 + n21 + n22
  accuracy = (n12 - n21) / total

  return f'Accuracy: {accuracy}, p-value: {p_value}'

##########################################################
# NEURAL NETWORKS
def hyperbolic_tangent(x):
  return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def rectified_linear_unit(x):
  return np.array([max(0,i) for i in x])

def logistic_sigmoid(x):
  return 1 / (1 + np.exp(-x))