from formulas import *
import numpy as np
import sklearn.linear_model as lm
from sklearn.linear_model import Ridge
from scipy.stats import multivariate_normal
from scipy.stats import norm
import matplotlib.pyplot as plt

##################################################################
print('QUESTION 3')
# All vectors in V has to be length = 1 and orthogonal

# A
v_13 = -0.1399; v_25 = -0.2116; s_22 = 5.1903

V3 = np.array([v_13, 0.0004, -0.7036, -0.1781, 0.5973])
V5 = np.array([0.6652, v_25, 0.0010, -0.1173, 0.3467])
print(f'A norms: V3={np.linalg.norm(V3)}, V5={np.linalg.norm(V5)}')

#B
v_13 = -0.3413; v_25 = -0.6508; s_22 = 26.9387
V3 = np.array([v_13, 0.0004, -0.7036, -0.1781, 0.5973])
V5 = np.array([0.6652, v_25, 0.0010, -0.1173, 0.3467])
print(f'B norms: V3={np.linalg.norm(V3)}, V5={np.linalg.norm(V5)}')
print(f'B orthogonality: {V3 @ V5}') # Very close to 0

#ANSWER IS B

#C
v_13 = 0.3425; v_25 = -0.6506; s_22 = 26.9387
V3 = np.array([v_13, 0.0004, -0.7036, -0.1781, 0.5973])
V5 = np.array([0.6652, v_25, 0.0010, -0.1173, 0.3467])
print(f'C norms: V3={np.linalg.norm(V3)}, V5={np.linalg.norm(V5)}')
print(f'C orthogonality: {V3 @ V5}')

#D
v_13 = -1.8385; v_25 = -0.1629; s_22 = 5.1903
V3 = np.array([v_13, 0.0004, -0.7036, -0.1781, 0.5973])
V5 = np.array([0.6652, v_25, 0.0010, -0.1173, 0.3467])
print(f'D norms: V3={np.linalg.norm(V3)}, V5={np.linalg.norm(V5)}')

print()

##################################################################
print('QUESTION 5')

print('relative density: ', relative_density([1.3, 2.4, 2.7]))

print()

##################################################################
print('QUESTION 12')

M_AB = np.array([
  [416, 42],
  [38, 68]
])
M_AC = np.array([
  [68, 38],
  [42, 416]
])

print(f'M_AB: {mcNemar(M_AB)}')
print(f'M_AC: {mcNemar(M_AC)}')

print()

##################################################################
print('QUESTION 19')

# y = np.array([-0.5, 0.39, 1.19, -1.08]).T
# x = np.array([[-0.86], [-0.61], [1.37], [0.1]])
# print(x)
# Lambda = 0.25

# w0 = 0.0
# w = np.array([0.39, 0.77]).T
# print(w)

# def f(x, w0, w):
#   return w0 + x.T @ w

# x_A = np.array([x, x**3]).T
# print('x_A',x_A)
# E_A = (y - f(x_A, w0, w))**2 + Lambda * (np.linalg.norm(w))**2
# print(E_A)
# print(f'E_A: {np.sum(E_A)}')

# x_C = np.array([x, np.sin(x)]).T
# E_C = (y - f(x_C, w0, w))**2 + Lambda * (np.linalg.norm(w))**2
# print(f'E_C: {np.sum(E_C)}')

# x_D = np.array([x, x**2]).T
# E_D = (y - f(x_D, w0, w))**2 + Lambda * (np.linalg.norm(w))**2
# print(f'E_D: {np.sum(E_D)}')

# print()

##################################################################
print('QUESTION 20')

W1 = np.array([
  [0, 0],
  [1, -2]
])
W2 = np.array([
  [0],
  [1],
  [1]
])

x = np.linspace(-0.5, 0.5, 20)
print(rectified_linear_unit(x))

# ones = np.ones(x)
# x_t = np.array([1, x])


print()

##################################################################
print('QUESTION 24')

w = [1/10 for i in range(10)]
correctly_classified = [0, 1, 1, 1, 1, 1, 0, 1, 1, 0]

print(adaboost(w, correctly_classified))

print()

##################################################################
print('QUESTION 25')



print()