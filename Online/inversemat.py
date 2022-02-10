# Python program to inverse
# a matrix using numpy

# Import required package
import numpy as np

# Taking a 4 * 4 matrix
A = np.array([[6, 1, 1, 3],
			[4, -2, 5, 1],
			[2, 8, 7, 6],
			[3, 1, 9, 7]])

# Calculating the inverse of the matrix
print(np.linalg.inv(A))
