import numpy as np

# Question 1
O1 = np.array([0, 4, 7, 9, 5, 5, 5, 6])
O3 = np.array([7, 7, 0, 10, 6, 6, 4, 9])

np.sqrt((O1**2).sum())

cos_o1o3 = (O1.T@O3) / (np.linalg.norm(O1) * np.linalg.norm(O3))
print(cos_o1o3)
