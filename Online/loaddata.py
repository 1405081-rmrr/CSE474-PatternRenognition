import numpy as np
import sys

matrix=[]
_count=0
def normal(mean,variance,value):
    x=pow((2*3.1416*variance),0.5)
    print("x ",x)
    y=1/x
    print("y ",y)
    epc=sys.float_info.epsilon
    print("Epsilon ",epc)
    a=pow((value-mean),2)
    a=a/(2*variance)
    print("a ",a)
    epc=pow(epc,a)
    y=y*epc
    print("y ",y)
    return y
    
with open('Test.txt') as f:
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
   # print(final_mat[:,2])
  #  print(sum(final_mat[:,2]))
    print("Mean of Column 1 and 2\n")
    mean=np.mean(final_mat,axis=0)
    var=np.var(final_mat,axis=0)
    print(mean)
    print("Variance of Column 1 and 2 \n")
    print(var)
    print(normal(110,2975,120))
    """for i in range(_count):
        for j in range(3):
            print(matrix[i][j],end=" ")
        print()
        """