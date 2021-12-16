import matplotlib.pyplot as plt
_list=list()
with open("Train.txt") as f:
    for line in f:
        for bl in line.split():
            bl=float(bl)
            _list.append(bl)
        del _list[-1]
    #print(_list)
x=list()
y=list()
for i in range(len(_list)):
    if(i%2==0):
        x.append(_list[i])
    else:
        y.append(_list[i])
print(x)
print(y)
"""
# x-axis values
x = [1,2,3,4,5,6,7,8,9,10]
# y-axis values
y = [2,4,5,7,6,8,9,11,12,12]
"""
# plotting points as a scatter plot
plt.scatter(x, y, label="stars", color= "blue",
			marker= "*", s=30)

# x-axis label
plt.xlabel('x - axis')
# frequency label
plt.ylabel('y - axis')
# plot title
plt.title('My scatter plot!')
# showing legend
plt.legend()

# function to show the plot
plt.show()
