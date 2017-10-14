import sys, math
import pandas as pd
import numpy as np
from random import seed
from random import random

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
# url = "Test2.csv"
input_dataset = pd.read_csv(url, header=None)
# Removing null or missing values
input_dataset = input_dataset.dropna()
# Converting categorical or nominal value to numerical values
for column in input_dataset.columns:
     input_dataset[column] = input_dataset[column].astype('category')
     input_dataset[column] = input_dataset[column].cat.codes
#print(input_dataset)

numberOfColumns = len(input_dataset.columns)
min = input_dataset[numberOfColumns-1].min()
max = input_dataset[numberOfColumns-1].max()

'''for index, row in input_dataset.iterrows():
    x = row[(len(row)-1)]
    if max-min != 0:
        print((x - min) / (max - min))
        row[(len(row) - 1)] = (x - min)/(max - min)'''

print(input_dataset)

def initialize_newnetwork(numberOfInputs, hiddenneuronsList, outputNeurons):
    network = list()
    n_inputs = numberOfInputs
    n_outputs = outputNeurons

    for n_hidden in hiddenneuronsList:
        hidden_layer = []
        for i in range(n_hidden):
            innerList = []
            for j in range(n_inputs + 1):
                innerList.append(random())
            hidden_layer.append(innerList)
        n_inputs = n_hidden
        network.append(hidden_layer)

    # print("Initialized Network: ", network)
    # return network
    output_layer = []
    for i in range(n_outputs):
        innerList = []
        for j in range(n_hidden + 1):
            innerList.append(random())
        output_layer.append(innerList)

    network.append(output_layer)

    print("Initialized Network: ", network)
    return network

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
     network = list()
     #hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
     hidden_layer = []
     for i in range(n_hidden):
        innerList = []
        for j in range(n_inputs + 1):
            innerList.append(random())
        hidden_layer.append(innerList)

     network.append(hidden_layer)
     #output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
     output_layer = []
     for i in range(n_outputs):
        innerList = []
        for j in range(n_hidden + 1):
            innerList.append(random())
        output_layer.append(innerList)

     network.append(output_layer)

     #print(network)
     return network

def findSigmoid(value):
    value = 1/(1 + math.exp(-1 * value))
    return round(value, 2)

def findNetList(layer, row):
    netLayerList = []
    for i in range(len(layer)):
        value = layer[i][0]  # bias
        for j in range(len(layer[i]) - 1):
            value = value + row[j] * layer[i][j + 1]
        # print(findSigmoid(value))
        netLayerList.append(findSigmoid(value))
    return netLayerList


def forwardPass(network, row):
    networkNetList = [row]
    netLayerList = row
    for layer in network:
        # print(layer)
        netLayerList = findNetList(layer, netLayerList)
        networkNetList.append(netLayerList)
    return networkNetList
        # layer = [[1, 2, 3], [1, 1, 1], [1, 1, 1]]
        # Iterate through each hidden layer node

def findDelta(layer, outputList, deltaList):
    newDeltaList = []
    # Update Bias weights
    for x in range(len(layer)):
        layer[x][0] = layer[x][0] + (0.5 * deltaList[x] * 1)

    # Update other weights
    k = 0
    for i in range(len(outputList)):
        value = 0
        k = k + 1
        for j in range(len(layer)):
            value = value + (layer[j][k] * deltaList[j])
            layer[j][k] = layer[j][k] + (0.5 * deltaList[j] * outputList[i])
        newDeltaList.append(value * outputList[i]*(1-outputList[i]))
    return newDeltaList

def backwardPass(network, networkNetList, row):
    length = len(networkNetList)
    targetIndex = len(row)-1
    #print(row[targetIndex])
    # Get the final output list
    outputList = networkNetList[length-1]
    deltaList = []
    for i in range(len(outputList)):
        deltaList.append(outputList[i] * (1 - outputList[i]) * (row[targetIndex] - outputList[i]))

    # Remove the last list of net values from networkNetList
    del networkNetList[-1]

    # Traverse through each layer in the network starting from last
    for layer in reversed(network):
        length = len(networkNetList)
        # print("Length of networkNetList %s" %(length))
        deltaList = findDelta(layer, networkNetList[length-1], deltaList)
        del networkNetList[-1]
    return

def findError(output,target):
    error = 0.5 * math.pow((target-output), 2)
    return error

def main():
    # training_percent = 0.8
    maxiterations = 10
    # numberofhiddenlayers = 2
    hiddenneuronsList = [4, 2]
    outputNeurons = 1
    # neuronsInEachLayer = [len(input_dataset.columns), hiddenneurons, outputNeurons]
    # network = initializeNetworkWeights(neuronsInEachLayer)
    # numberofneurons = []
    # for i in numberofhiddenlayers:
    #     numberofneurons[i] = sys.argv[0]
    seed(1)
    numberOfInputs = len(input_dataset.columns) -1   # All except target value
    # network = initialize_network(numberOfInputs, 3, 2) # parameters: Number of inputs, Number of neurons in hidden layer, Number of outputs
    network = initialize_newnetwork(numberOfInputs, hiddenneuronsList, outputNeurons)
    # print(" Network weights: ")
    # print(network)
    error = -1
    while error != 0 and maxiterations != 0:
        for index, row in input_dataset.iterrows():
            networkNetList = forwardPass(network, row[:-1])
            n = len(networkNetList)
            finalOutputList = networkNetList[n-1]
            # print(networkNetList)
            # print("Foward pass weights: ",network)
            backwardPass(network, networkNetList, row)
            # print("Backward pass weights: ",network)
            # print("Target", row[(len(row)-1)])
            x = row[(len(row)-1)]
            target = (x - min) / (max - min)
            error = findError(finalOutputList[0],target)
            #print("weights: ", network)
            print("Error ",error)
        maxiterations = maxiterations - 1

main()




