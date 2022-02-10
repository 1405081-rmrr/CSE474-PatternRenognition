import numpy as np
import pandas as pd
import sklearn

class Perceptron:
    def __init__(self, learning_rate=0.01, iteration=100):
        self.lr = learning_rate
        self.iteration = iteration
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        number_of_samples, number_of_features = X.shape

        # init parameters
        self.weights = np.zeros(number_of_features)
        self.bias = 0
        print("Hahah ",y)
        y_ = np.array(y)
        print("Actual y ",y_)

        for _ in range(self.iteration):

            for idx, x_i in enumerate(X):

                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                #print("In Loop ",y_predicted)
                # Perceptron update rule
                update = self.lr * (y_[idx] - y_predicted)

                self.weights += update * x_i
                self.bias += update
                #print(linear_output)
                
            

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        print("Y Pred ",y_predicted)
        return y_predicted

    def _unit_step_func(self, x):
        #print("Inside unit ",x)
        return np.where(x <=0,1,2)
if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    
    def fileRead():
        feature=list()
        label=list()
        flag=0
        f=open("trainLinearlySeparable.txt")
        for line in f:
            info=list()
            x=line.split()
            if(flag==0):
                for item in x:
                    item=float(item)
                    info.append(item)
                flag=1
            elif(flag==1):
                for i in range(0,len(x)-1):
                    item=float(x[i])
                    info.append(item)
                feature.append(info)
                label.append(float(x[len(x)-1]))
        X=np.array(feature)
        y=np.array(label)
        f.close()
        feature=list()
        label=list()
        flag=0
        f=open("testLinearlySeparable.txt")
        for line in f:
            info=list()
            x=line.split()
            for i in range(0,len(x)-1):
                item=float(x[i])
                info.append(item)
            feature.append(info)
            label.append(float(x[len(x)-1]))
        X_test=np.array(feature)
        y_test=np.array(label)
        f.close()
        X_train, X_test, y_train, y_test =(X,X_test,y,y_test)
        print(X_train)
        return X_train,y_train,X_test,y_test
                
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    X_train,y_train,X_test,y_test=fileRead()
    p = Perceptron(learning_rate=0.01, iteration=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)
    
    print("Perceptron classification accuracy", accuracy(y_test, predictions))

        