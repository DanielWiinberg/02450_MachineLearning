from formulas import *
import numpy as np
import sklearn.linear_model as lm
from sklearn.linear_model import Ridge
from scipy.stats import multivariate_normal
from scipy.stats import norm
import matplotlib.pyplot as plt
import math

##################################################################
print('QUESTION 3')

pcs = [43.4, 23.39, 18.26, 9.34, 2.14]
print('3 pcs:', varianceExplained(pcs, 3))
print('4 pcs:', varianceExplained(pcs, 4))
print('1 pcs:', varianceExplained(pcs, 1))

print() 

##################################################################
print('QUESTION 8')

o5 = [24.2, 37.7, 38.5]
print(f'o5 density: {relative_density(o5)}')

print()

##################################################################
print('QUESTION 9')

o11 = [24.2, 39.4]
gaussian_KDE(o11, 20)

print()

##################################################################
print('QUESTION 12')

root = [2, 5, 4]
v1 = [2, 4, 0]
v2 = [1, 0, 4]
print(gini(root), gini(v1), gini(v2), purityGainGini(root, v1, v2))

print()

##################################################################
print('QUESTION 15')

mu_1 = (38.0 + 26.8) / 2
mu_2 = (15.1 + 12.8) / 2
s = 400

pdf_32 = norm.pdf(32, mu_1, np.sqrt(s))
pdf_14 = norm.pdf(14, mu_2, np.sqrt(s))
print(pdf_32, pdf_14)

print(f'Probability is: {(pdf_32 * pdf_14 * (2/11)) / 0.00010141}')

print()

##################################################################
print('QUESTION 19')

correctly_classified = np.array([[1 for i in range(429)] + [0 for i in range(572-429)]])
weights = np.array([1/572 for _ in range(572)])

# print(adaboost(weights, correctly_classified))

print()

##################################################################
print('QUESTION 21')

x = np.array([1, 2, 3, 4])
y = np.array([6, 2, 3, 4])

X = np.array([np.cos((np.pi * x)/ 2), np.sin((np.pi * x) / 2)]).T
print('transformed input', X)

# Solution to w* 14.2
w = np.linalg.inv(X.T @ X) @ X.T @ y
print('w', w)

print()

##################################################################
print('QUESTION 22')
x = np.array([1, 2, 3, 4])
y = np.array([6, 2, 3, 4])

X = (x - np.mean(x) / standard_deviation(x))

w = -np.sqrt(3/20)
w0 = 15/4
print(np.linalg.norm(y - w0 - X - w)**2)


##################################################################
print('QUESTION 25')

units = 200000
def training(k):
  n = math.ceil((800) - (800/k))
  return k*3*(n * np.log2(n) + (800 - n)) + 800*np.log2(800) + 200

print('K=7', training(7))
print('K=8', training(8))
print('K=9', training(9))
print('K=10', training(10))