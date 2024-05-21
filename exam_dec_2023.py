from formulas import *
import numpy as np
import sklearn.linear_model as lm
from sklearn.linear_model import Ridge
from scipy.stats import multivariate_normal
from scipy.stats import norm
import matplotlib.pyplot as plt

################################################################
#Question 3
x4 = np.array([1.4, 1, 0.1, -2.8]).T
X = np.array([
  [-0.6, -0.6, 2.5, -0.1],
  [-0.8, -0.3, -1, 1.2],
  [-0.7, 0.3, -0.2, -0.1],
  [1.4, 1, 0.1, -2.8],
  [-0.2, 0.8, -1.2, 0.7]
])
E = np.array([
  [3.7, 0, 0, 0],
  [0, 3.04, 0, 0],
  [0, 0, 0.56, 0],
  [0, 0, 0, 0]
])
V = np.array([
  [0.43, -0.26, 0.22, -0.84],
  [0.17, -0.37, 0.81, 0.42],
  [0.33, 0.88, 0.34, -0.01],
  [-0.82, 0.14, 0.42, -0.36]
])
# Playing around with matrix reversing
print('V @ V.T')
print(V @ V.T)
print('V @ V^-1')
print(V @ np.linalg.inv(V))

# First we compute the mean of X
X_mean = np.mean(X, axis=0) # If axis=0 is omitted it calculates the mean of the entire matrix, not column-wise
print('X_mean', X_mean)

# Then we center the matrix by subtracting the mean from all values in X
X_centered = X - X_mean
x4_centered = x4 - X_mean #x4_centered = X_centered[3]

V2 = V[:, :2] # First two principal components

# Project x4 on the first two principal components
x4_proj = x4_centered.T @ V2
print('x4_proj', x4_proj) # Will be a 1X2 matrix, which is the x4 projections values projected onto the first 2 principal components

x4_reconstructed = (x4_proj @ V2.T) + X_mean # Reversing the projection step, but only using the first two principal components
print('x4_reconstructed', x4_reconstructed)
print()

x4_reconstructed_in_one_step = X_mean + V2@V2.T @ x4_centered
print('x4_reconstructed one step', x4_reconstructed_in_one_step)
print()

################################################################
#Question 4
pcs = [3.7, 3.04, 0.56, 0.48]
# print('two vectors:', varianceExplained(pcs, 2))

################################################################
#Question 6

def transform_A(b1, b2):
    return np.array([1, b1**2, b2**3])
def transform_B(b1, b2):
    return np.array([1, b1**2, b2**3])
def transform_C(b1, b2):
    return np.array([1, b1**3, b2**2])
def transform_D(b1, b2):
    return np.array([1, b1**3, b2**2])

b1_values = np.linspace(-4, 4, 100)
b2_values = np.linspace(-4, 4, 100)
B1, B2 = np.meshgrid(b1_values, b2_values)

wA = np.array([0.31, -0.06, 0.07]).T
wB = np.array([0.72, 3.13, -0.25]).T
wC = np.array([0.31, -0.06, 0.07]).T
wD = np.array([0.72, 3.13, -0.25]).T

# Apply the nonlinear transformation and compute the classifier's output
Z = np.zeros_like(B1)
for i in range(B1.shape[0]):
  for j in range(B1.shape[1]):
    b_transformed = transform_D(B1[i, j], B2[i, j])
    Z[i, j] = sigmoid(np.dot(wD, b_transformed))

# Plot the decision boundary
plt.figure(figsize=(10, 8))
contour = plt.contourf(B1, B2, Z, levels=10, cmap='viridis')
cbar = plt.colorbar(contour, label='Probability of class High')
# plt.contour(B1, B2, Z, levels=[0.5], colors='red', linewidths=2)
plt.title('Probability Surface and Decision Boundary')
plt.xlabel('b1')
plt.ylabel('b2')
plt.grid(True)
# plt.show()

# OPTION D IS CORRECT

# Calculation using input
b1, b2 = 0, -2

wA = np.array([0.31, -0.06, 0.07]).T
wB = np.array([0.72, 3.13, -0.25]).T
wC = np.array([0.31, -0.06, 0.07]).T
wD = np.array([0.72, 3.13, -0.25]).T

bA = np.array([1, b1**2, b2**3]).T
bB = np.array([1, b1**2, b2**3]).T
bC = np.array([1, b1**3, b2**2]).T
bD = np.array([1, b1**3, b2**2]).T

pA = sigmoid(wA @ bA.T)
pB = sigmoid(wB @ bB.T)
pC = sigmoid(wC @ bC.T)
pD = sigmoid(wD @ bD.T)

# print('pA', pA) # Not A
# print('pB', pB) # Not B
# print('pC', pC) # Not C
print('pD', pD)

################################################################
#Question 7
print('QUESTION 7')
w02 = 2.2
w2 = np.array([-0.7, 0.5]).T
w11 = np.array([2.2, 0.7, -0.3])
w12 = np.array([-0.2, 0.8, 0.4])
print(w11, w12)

x1, x2 = -2.0, -1.88

n1 = hyperbolic_tangent(np.array([1, x1, x2] @ w11.T))
n2 = hyperbolic_tangent(np.array([1, x1, x2] @ w12.T))
print(f'activation hidden neurons, n1: {n1}, n2: {n2}')

fxw = hyperbolic_tangent(w02 + w2[0] * n1 + w2[1] * n2)
print('fxw', fxw)
#fxw =  0.798
# ANSWER IS C
print()

################################################################
#Question 12
print('QUESTION 12')
y = np.array([-2.9, -0.4, 0.7, 2.5, 4.5])
x = np.array([-3.4, -1.3, 0.5, 2.4, 4.2])
Lambda = 0.7

#standardized
x = (x - np.mean(x)) / standard_deviation(x)

#Creating a matrix with ones in the first column
X = np.vstack([[1, 1, 1, 1, 1], x]).T

I = np.eye(X.shape[1])
I[0, 0] = 0  # Do not regularize the intercept term

w = np.linalg.inv(X.T @ X + Lambda * I) @ X.T @ y #ridge regression weights w = (X⊤*X + λ*I)−1 X⊤*y
print('weights: ', w) 

y2_pred_1 = w[0] + w[1]*x[1]
# y2_pred_1 = w[0] + 2.3855*x[1]

#Using sklearn
ridge_model = Ridge(alpha=Lambda, fit_intercept=True) #If fit_intercept is set to False, it will apply the regularization to the intercept term as well
ridge_model.fit(X, y)

X2 = np.array([1, x[1]]).reshape(1, -1)
y2_pred = ridge_model.predict(X2)

print('y2_pred manual = ', y2_pred_1)
print('y2_pred sklearn = ', y2_pred)
print()

################################################################
#Question 13
print('QUESTION 13')

p_1 = 0.53
p_2 = 0.47

mu_1 = np.array([0.77, -0.41])
E_1 = np.array([
   [0.29, -0.12],
   [-0.12, 0.55]
])

mu_2 = np.array([-0.91, 0.5])
E_2 = np.array([
   [0.32, -0.11],
   [-0.11, 1.12]
])

x_test = np.array([0.0, 0.7]).T

# Create multivariate normal distributions
dist_1 = multivariate_normal(mean=mu_1, cov=E_1)
dist_2 = multivariate_normal(mean=mu_2, cov=E_2)

# Calculate the PDF values at the test point
pdf_1 = dist_1.pdf(x_test)
pdf_2 = dist_2.pdf(x_test)

print(f'pdf_1: {pdf_1}')
print(f'pdf_2: {pdf_2}')

# Calculate the unnormalized posterior probabilities
unnormalized_posterior_1 = pdf_1 * p_1
unnormalized_posterior_2 = pdf_2 * p_2

# Normalize the posteriors so they sum to 1
total = unnormalized_posterior_1 + unnormalized_posterior_2
posterior_1 = unnormalized_posterior_1 / total
posterior_2 = unnormalized_posterior_2 / total

print(f"Posterior probability for class y=1: {posterior_1}")
print(f"Posterior probability for class y=2: {posterior_2}")

################
# NAIVE BAYES
# Extract the means and variances for each feature
mu_1_f1, mu_1_f2 = mu_1
var_1_f1, var_1_f2 = E_1[0, 0], E_1[1, 1]

mu_2_f1, mu_2_f2 = mu_2
var_2_f1, var_2_f2 = E_2[0, 0], E_2[1, 1]

pdf_1_f1 = norm(mu_1_f1, np.sqrt(var_1_f1)).pdf(x_test[0])
pdf_1_f2 = norm(mu_1_f2, np.sqrt(var_1_f2)).pdf(x_test[1])
likelihood_1 = pdf_1_f1 * pdf_1_f2

pdf_2_f1 = norm(mu_2_f1, np.sqrt(var_2_f1)).pdf(x_test[0])
pdf_2_f2 = norm(mu_2_f2, np.sqrt(var_2_f2)).pdf(x_test[1])
likelihood_2 = pdf_2_f1 * pdf_2_f2

# Calculate the unnormalized posterior probabilities
unnormalized_posterior_1 = likelihood_1 * p_1
unnormalized_posterior_2 = likelihood_2 * p_2

# Normalize the posteriors so they sum to 1
total = unnormalized_posterior_1 + unnormalized_posterior_2
posterior_1 = unnormalized_posterior_1 / total
posterior_2 = unnormalized_posterior_2 / total

print(f"Posterior probability for class y=1: {posterior_1}")
print(f"Posterior probability for class y=2: {posterior_2}")

#PROBILITY COMPLETELY FLIPPED WHEN USING NAIVE BAYES
# The features are very correlated, can be seem from the E matrix. That means the naive bayes assumption that they are not correlated is not very good. 
# The task was to use naive bayes though, so then the correct answer is that the point should be assigned to class 2 => Answer C

print()

################################################################
#Question 18

weights = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
classified_correct = [1, 1, 0, 0, 0, 0]
w1 = adaboost(weights, classified_correct)
print('weights after first round: ', w1)
w2 = adaboost(w1, classified_correct)
print('weights after second round: ', w2)

################################################################
#Question 21
anomalies = detect_anomalies_kde([-3, -1, 5, 6], [-4, 2], threshold=0.015)
print('Question 21: ', anomalies)