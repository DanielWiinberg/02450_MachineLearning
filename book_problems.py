from formulas import *

######################################################################
# CHAPTER 3
# Question 3.2
pca3_2 = [17, 15.2, 13.1, 13.0, 11.8, 11.3]
print('Question 3.2: ', varianceExplained(pca3_2, 3))
print()

# Question 3.3
pca_3_3 = [9.7, 6.7, 5.7, 3.7, 3.0, 1.3, 0.7]
print('Question 3.3 a: ', varianceExplained(pca_3_3, 1))
print('Question 3.3 b: ', varianceExplained(pca_3_3, 3))
print('Question 3.3 c: ', 1 - varianceExplained(pca_3_3, 6))
print('Question 3.3 5: ', varianceExplained(pca_3_3, 5))
print()

# Question 3.5
v = np.array( [[-0.99, -0.13, -0.00],
              [-0.09, 0.70, -0.71],
              [0.09, -0.70, -0.71]])
x = np.array([3-7/3, 2-4/3, 1-5/3]) #Mean is subtracted in this case
print(np.dot(x, v))
print()

# Question 3.6
pca_3_6 = [2.69, 2.53, 1.05, 0.83, 0.49, 0.31]
print('Question 3.6a: ', varianceExplained(pca_3_6, 1))
print('Question 3.6b: ', varianceExplained(pca_3_6, 3))
print('Question 3.6c: ', 1 - varianceExplained(pca_3_6, 4))
print()


######################################################################
# CHAPTER 8

# Question 8.4
conf_LR = [12, 69, 10, 215] #[TP, FN, FP, TN]
conf_DT = [26, 65, 34, 191]
print('Logistic')
calculateConfMatrixStats(conf_LR)
print('Decision tree')
calculateConfMatrixStats(conf_DT)

######################################################################
# CHAPTER 9

# Question 9.1
root = [225, 81]
v1 = [62, 170-62]
v2 = [19, 136-19]
print(gini(root), gini(v1), gini(v2), purityGainGini(root, v1, v2))

######################################################################
# CHAPTER 10

# Question 10.5
conf_LR = [12, 69, 10, 215]
conf_DT = [26, 55, 34, 191]
print('Question 10.5')
print('Logistic regression')
calculateConfMatrixStats(conf_LR)
print('Decision tree')
calculateConfMatrixStats(conf_DT)

######################################################################
# CHAPTER 20

# Question 20.1
o8_distance = np.array([5.11, 4.79, 4.9, 2.96, 5.16, 2.88])
gaussian_KDE(o8_distance)

# Question 20.5
o8 = [2.88, 2.96, 4.74]
neighbours_o8 = [[0.96, 1.76, 1.52], [1.41, 2.66, 2.84], [1.41, 2.88, 3.54]]
print('ard: ', average_relative_density(o8, neighbours_o8))

#Question 20.6
o1 = [4]
neighbour_o1 = [[3]]
print('ard: ', average_relative_density(o1, neighbour_o1))