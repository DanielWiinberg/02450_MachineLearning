# exercise 2.1.2

# Imports the numpy and xlrd package, then runs the ex2_1_1 code
from ex2_1_1 import *
# (requires data structures from ex. 2.1.1)

from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show

# Data attributes to be plotted
A = 0
B = 1

##
# Make a simple plot of the i'th attribute against the j'th attribute
# Notice that X is of matrix type (but it will also work with a numpy array)
# X = np.array(X) #Try to uncomment this line
plot(X[:, A], X[:, B], 'o')

# %%
# Make another more fancy plot that includes legend, class labels, 
# attribute names, and a title.
f = figure()
title('NanoNose data')

for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(X[class_mask,A], X[class_mask,B], 'o',alpha=.3)

legend(classNames)
xlabel(attributeNames[A])
ylabel(attributeNames[B])

# Output result to screen
show()
print('Ran Exercise 2.1.2')