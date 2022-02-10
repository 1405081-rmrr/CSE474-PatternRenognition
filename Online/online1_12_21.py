import numpy as np
import sys
from numpy.linalg import inv
from numpy.lib.function_base import cov

matrix=[]
_count=0
class1_matrix=[]
class2_matrix=[]

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
   
def multivariant(covariance,mean,matrix_class):
    calculatepi=pow((2*3.1416),2/2)
    cal_covariance=np.linalg.det(covariance)
    cal_covariance=pow(cal_covariance,0.5)
    calculatepi=calculatepi*cal_covariance
    calculatepi=1/calculatepi
    epc=sys.float_info.epsilon
    mat_cal=(matrix_class-mean)
    mat_caltrans=np.transpose(mat_cal)
    cov_calc=np.linalg.inv(covariance)
    second_mul=mat_caltrans*cov_calc
    second_mul=second_mul*mat_cal
    second_part=pow(epc,second_mul)
    final_result=calculatepi*second_part
    return final_result
   
with open('/media/roktim/PDF/BUET-LEVEL-TERM/L-4 T-2/CSE474/Online/Train.txt') as f:
    for line in f:
        a=list()
        _count+=1
        for item in line.split():
            #print(item," ")
            item=float(item)
            a.append(item)
        if(a[2]==1):
            demo=[]
            demo.append(a[0])
            demo.append(a[1])
            class1_matrix.append(demo)
        elif(a[2]==2):
            demo2=[]
            demo2.append(a[0])
            demo2.append(a[1])
            class2_matrix.append(demo2)
    print("Number of rows ",_count)
    final_mat1=np.array(class1_matrix)
    final_mat2=np.array(class2_matrix)
    final_mat1_tranpose=np.transpose(final_mat1)
    final_mat2_transpose=np.transpose(final_mat2)

   # print(final_mat)
    print("Class 1 Mat\n ",final_mat1)
    print("Class 2 Mat\n ",final_mat2)

   # print(final_mat[:,2])
  #  print(sum(final_mat[:,2]))
    mean=np.mean(final_mat1,axis=0)
    print("Mean value of Class1\n")
    print(mean)
    mean2=np.mean(final_mat2,axis=0)
    print("Mean value of Class2\n")
    print(mean2)
    cov_class1=[]
    cov_class2=[]
    cov_class1=np.cov(final_mat1_tranpose)
    cov_class2=np.cov(final_mat2_transpose)
    print("Cov of class 1 \n")
    print((cov_class1))
    print("Cov of class 2 \n")
    print((cov_class2))
    mean1_np=np.array(mean)
    mean2_np=np.array(mean2)
    feature_mean1=np.subtract(final_mat1,mean1_np)
    feature_mean_transpose1=np.transpose(feature_mean1)
    print("After transpose\n ",feature_mean_transpose1)
    print(feature_mean_transpose1.shape)
    cov_class1_inv=inv(cov_class1)
    feature_inverse_cov1=np.dot(feature_mean1,cov_class1_inv)
    feature_inverse_cov1=np.transpose(feature_inverse_cov1)
    print("First Part\n",feature_inverse_cov1)
    bracket=np.dot(feature_inverse_cov1,feature_mean1)
    print("Bracket\n",bracket)

    
#এখানে একটা ভুল করছিলাম। যেহেতু ফিচার ২ টা কলাম অনুযায়ী আছে তাই আগে ম্যাট্রিক্সকে transpose বানাতে
#হবে। এরপর এর covariance বের করতে হবে। 
# transpose বানালে রোতে ফিচার গুলা থাকবে। তাই covariance বের করলে হবে 2x2 ম্যাট্রিক্স।
# n টা ফিচার থাকলে covariance ম্যাট্রিক্স হবে nXn।
#2-12-2021