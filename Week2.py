import numpy as np
import xlrd
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show
from scipy.linalg import svd

################################################
# Read excel document
dataPath = 'C:/Users/danwii/Documents/02450_MachineLearning/02450Toolbox_Python/Data/nanonose.xls'
doc = xlrd.open_workbook(dataPath).sheet_by_index(0)

attributeNames = doc.row_values(0, 3, 11)

classLabels = doc.col_values(0, 2, 92)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(5)))
# print(classNames, attributeNames)

y = np.asarray([classDict[value] for value in classLabels]) # y is classNames encoded as values from 0-4
# print(y)

X = np.empty((90, 8)) # X is 90 rows of length 8, this is the data
for i, col_id in enumerate(range(3, 11)):
  # print(i, col_id)
  X[:, i] = np.asarray(doc.col_values(col_id, 2, 92))

################################################
# PLOTTING

basicPlot = figure('Basic Plot')
A = X[:,0]
B = X[:,1]
plot(A, B, 'o')

################################################
# PCA
standardized = X - np.ones((len(y),1))*X.mean(axis=0)

U,S,Vh = svd(standardized, full_matrices=False) 
V = Vh.T # SciPy apparently return V as tranposed, therefore tranpose back

# Explained variance
rho = (S*S) / (S*S).sum()
print(rho)

threshold = 0.9

explainedVarianceFigure = figure('Explained variance')
plot(range(1,len(rho)+1),rho,'x-') # Explained variance by single PC
plot(range(1,len(rho)+1),np.cumsum(rho),'o-') # Cumulated values
plot([1,len(rho)],[threshold, threshold],'k--')
title('Explained variance')

# DEW: From the plot it can be seen that using three principal components
# we exceed the limit of 90%. We just need 4 to exceed 95%. 

################################################
# PLOT PRINCIPAL COMPONENTS
Z = standardized @ V #TODO: Why this?

pca_figure = figure('PCA')
title('PCA')

for i in range(len(classNames)):
  index_of_class = i == y
  plot(Z[index_of_class, 0], Z[index_of_class, 1], 'o')

title('Principal components')
xlabel('PCA1')
ylabel('PCA2')

#TODO: Not sure what the benefits are in this case

################################################
# PLOT PRINCIPAL COMPONENTS

################################################
# SHOW AT THE END
show()