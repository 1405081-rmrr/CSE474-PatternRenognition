# Python Program illustrating
# numpy.var() method
import numpy as np
matrix=[]
_count=0
with open('Trial.txt') as f:
    for line in f:
        a=list()
        _count+=1
        for item in line.split():
            #print(item," ")
            item=float(item)
            a.append(item)
        matrix.append(a)
    print(_count)
    final_mat=np.array(matrix)
    print(final_mat)
	
# 2D array
"""arr = [[2, 2, 2, 2, 2],
	[15, 6, 27, 8, 2],
	[23, 2, 54, 1, 2, ],
	[11, 44, 34, 7, 2]]
"""
arr=final_mat
	
# var of the flattened array
	
# var along the axis = 0
print("\nvar of arr, axis = 0 : ", np.var(arr, axis = 0))

# var along the axis = 1
print("\nmean of arr, axis = 0 : ", np.mean(arr, axis = 0))