from formulas import *
import numpy as np
import sklearn.linear_model as lm
from sklearn.linear_model import Ridge
from scipy.stats import multivariate_normal
from scipy.stats import norm
import matplotlib.pyplot as plt
import math

##################################################################
print('QUESTION 5')

pcs = [30.19, 16.08, 11.07, 5.98]

print(f'1pc: {varianceExplained(pcs, 1)}')
print(f'2pcs: {varianceExplained(pcs, 2)}')
print(f'3pcs: {varianceExplained(pcs, 3)}')
print(f'last pc: {1 - varianceExplained(pcs, 3)}')

print()

##################################################################
print('QUESTION 7')

x = np.array([-1, -1, -1, 1])

V = np.array([
  [0.45, -0.60, -0.64, 0.15],
  [-0.40, -0.80, 0.43, -0.16],
  [0.58, -0.01, 0.24, -0.78],
  [0.55, -0.08, 0.59, 0.58]
])

print('x projected onto principal components')
print('x @ V14')
V14 = V[:, [0, 3]]
print(x @ V14)

print('x @ V24')
V24 = V[:, [1, 3]]
print(x @ V24)

print('x @ V23')
V23 = V[:, [1, 2]]
print(x @ V23)

print('x @ V34')
V34 = V[:, [2, 3]]
print(x @ V34)

print()

##################################################################
print('QUESTION 8')

zq_matr = np.array([
  [114, 0, 32],
  [0, 119, 0],
  [8, 0, 60]
])

print(rand_jaccard_index(zq_matr))

print()

##################################################################
print('QUESTION 12')

o2 = [75, 125]
o2_neigh = [[75, 51], [51, 125]]

print(f'ard: {average_relative_density(o2, o2_neigh)}')

##################################################################
print('QUESTION 18')

root = [146, 119, 68]
v1 = [146, 119, 0]
v2 = [0, 0, 68]

print(f'purity gain: {purityGainClassError(root, v1, v2)}')

print()

##################################################################
print('QUESTION 19')

N = 7
weights = [1/N for _ in range(N)]
correctly_classified = [1, 1, 0, 1, 1, 1, 1]
print(adaboost(weights, correctly_classified))

print()

##################################################################
print('QUESTION 21')

wA = np.array([423.49, 48.16]).T
wB = np.array([0.0, -46.21]).T
wC = np.array([0.0, -27.89]).T
wD = np.array([418.94, -26.12]).T

x2 = np.array([1, 16])

yA = sigmoid(x2.T @ wA)
yB = sigmoid(x2.T @ wB)
yC = sigmoid(x2.T @ wC)
yD = sigmoid(x2.T @ wD)

print(f'A: {yA}')
print(f'B: {yB}')
print(f'C: {yC}')
print(f'D: {yD}')

print()

##################################################################
print('QUESTION 22')

x = 15.38

w1 = 0.13; mu1 = 18.347; s1 = 1.2193
w2 = 0.55; mu2 = 14.997; s2 = 0.986
w3 = 0.32; mu3 = 18.421; s3 = 1.1354

pdf1 = norm.pdf(x, mu1, s1)
pdf2 = norm.pdf(x, mu2, s2)
pdf3 = norm.pdf(x, mu3, s3)

p_sum = w1*pdf1 + w2*pdf2 + w3*pdf3
print(f'prob: {p_sum}')

p_k2 = (w2*pdf2) / p_sum
print(f'p(k=2 | x0 = 15.38) = {p_k2}')

print()

##################################################################
print('QUESTION 23')

fold_1 = np.array([
  [86, 8],
  [7, 10]
])

fold_2 = np.array([
  [65, 15],
  [11, 20]
])

fold_3 = np.array([
  [79, 5],
  [17, 10]
])

print(f'fold_1: {mcNemar(fold_1)}')
print(f'fold_2: {mcNemar(fold_2)}')
print(f'fold_3: {mcNemar(fold_3)}')