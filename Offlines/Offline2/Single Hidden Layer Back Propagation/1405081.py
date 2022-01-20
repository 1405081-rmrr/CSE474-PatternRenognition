# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from random import random
from math import exp 

# Load our text file
def load_txt(filename):
    full_data=list()
    for line in filename:
        lines=list()
        for item in line.split():
            lines.append(item)
        full_data.append(lines)
    #print(full_data)
    return full_data
#Load CSV

def load_csv(filename):
    full_data=list()
    for line in filename:
        lines=list()
        for item in line.split(','):
            lines.append(item)
        full_data.append(lines)
    return full_data   
def str_to_float(dataset):
    new_dataset=list()
    for _list in dataset:
        temp=list()
        new_temp=list()
        temp=_list
        for i in range(len(temp)-1):
            temp[i]=float(temp[i])
            new_temp.append(temp[i])
        temp[len(temp)-1]=int(temp[len(temp)-1])
        new_temp.append(temp[len(temp)-1])
        new_dataset.append(new_temp)
    dataset.clear()
    dataset=list()
    dataset=new_dataset
    return dataset
 

def dataset_minmax(dataset):
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats
 

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
 

def evaluate_algorithm(dataset,dataset2 ,back_propagation,  *args):
    _class = list()
    actual=list()
    for _l in dataset2:
        actual.append(_l[len(_l)-1]) #testfile er actual class retreive korchi
        _l[len(_l)-1]=None
    predicted = back_propagation(dataset, dataset2, *args) #predicted class bair krchi
    """"for (i,j) in zip(predicted,actual):
        if(i==j):
            print("Match ",i," ",j)
        if(i!=j):
            print("Mismatch ",i," ",j)
            """
    accuracy = accuracy_metric(actual, predicted)
    _class.append(accuracy)
    return _class
 
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1 #number of feature hocche inputs.
    n_outputs = len(set([row[-1] for row in train])) #koydhoroner class ache
    n_outputs+=1
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs) #+1 deya lagbe
    predictions = list()
    for row in test:
        prediction=predict(network,row)
        predictions.append(prediction)
    return predictions

def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    #network.append(hidden_layer)
    #network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        for row in train: #train kora hocche
            #print("Row ",row[-1])
            outputs = forward_propagate(network, row) #forward propagation
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1 #Assume
            backward_propagate_error(network, expected) #back propagation error
            update_weights(network, row, l_rate) #update weights


def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation
 

def transfer(activation):
    return 1.0 / (1.0 + exp(-activation)) #Sigmoid function
 

 
#
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
# 
def transfer_derivative(output):
    return output * (1.0 - output)
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= l_rate * neuron['delta']
    #print(neuron)
#

# 


 

def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))
 


#seed(1)
with open('./trainNN.txt') as filename:
    
    dataset=load_txt(filename)
#for item in dataset:
    #print(item[len(item)-1],end=" ")
#print('\n')

n_hidden=1
output=open('./layer.txt','w')
while(n_hidden!=6):
    with open('./testNN.txt') as f:
        dataset2=load_txt(f)


    dataset2=str_to_float(dataset2)
    dataset=str_to_float(dataset)

    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)

    minmax = dataset_minmax(dataset2)
    normalize_dataset(dataset2, minmax)
    l_rate = 0.2
    n_epoch = 500
    #n_hidden = 3

    class_find = evaluate_algorithm(dataset, dataset2,back_propagation, l_rate, n_epoch, n_hidden)
    accu=sum(class_find)/float(len(class_find))
    print('Accuracy: %.3f%%' % (sum(class_find)/float(len(class_find))))
    n_hidden=str(n_hidden)
    out=""
    out+="Number of hidden Layer "+n_hidden+" "+"Accuracy "+" "+str(accu)+'\n'
    output.write(out)
    n_hidden=int(n_hidden)
    n_hidden+=1
    