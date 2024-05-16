from formulas import *

v = np.array ([
  [0.11, -0.8, 0.3, -0.17, -0.48],
  [-0.58, -0.31, 0.01, -0.5, 0.56],
  [0.49, 0.08, -0.49, -0.72, -0.07],
  [0.6, -0.36, 0.04, 0.27, 0.66],
  [-0.23, -0.36, -0.82, 0.37, -0.09]
])
x = np.array([15.5-12.9, 59.2-58.2, 1.4-1.7, 1438-1436.8, 5.3-4.1])

# QUESTION 4
print('2021 Question 4')
print(np.dot(x, v))
print()

# Question 5
x2 = np.array([39, 415, -7, -6727, 143])
x3 = np.array([0, -7, 1, 108, -2])
calculateCovarianceAndCorrelation(x2, x3)
print()