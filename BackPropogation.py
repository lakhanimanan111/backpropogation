import sys, math
import pandas as pd
import numpy as np
from random import seed
from random import random

filePath = sys.argv[1] if len(sys.argv) > 1 else "ProcessedFile"
input_dataset = pd.read_csv(filePath)
#print(input_dataset)
'''
# Removing null or missing values
input_dataset = input_dataset.dropna()
# Converting categorical or nominal value to numerical values
for column in input_dataset.columns:
    if input_dataset[column].dtype != np.number:
         input_dataset[column] = input_dataset[column].astype('category')
         input_dataset[column] = input_dataset[column].cat.codes

# Normalize the data
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

# Normalize the data
input_dataset = normalize(input_dataset)
'''


# Initialize random weights for entire network
def initialize_newnetwork(numberOfInputs, hiddenneuronsList, outputNeurons):
    network = list()
    n_inputs = numberOfInputs
    n_outputs = outputNeurons

    # Hidden layer weight initialization
    for n_hidden in hiddenneuronsList:
        hidden_layer = []
        for i in range(n_hidden):
            innerList = []
            for j in range(n_inputs + 1):
                innerList.append(random())
            hidden_layer.append(innerList)
        n_inputs = n_hidden
        network.append(hidden_layer)
    n_hidden = hiddenneuronsList[len(hiddenneuronsList)-1]

    # Output layer weight initialization
    output_layer = []
    for i in range(n_outputs):
        innerList = []
        for j in range(n_hidden + 1):
            innerList.append(random())
        output_layer.append(innerList)

    network.append(output_layer)

    return network


def findSigmoid(value):
     value = 1/(1 + math.exp(-1 * value))
     return round(value, 2)

'''def findSigmoid(value):
    if value < 0:
        return 1 - 1 / (1 + math.exp(value))
    return 1 / (1 + math.exp(-value))'''


# This method calculates net output for a layer in the network
def findNetList(layer, row):
    netLayerList = []
    for i in range(len(layer)):
        value = layer[i][0]  # bias
        for j in range(len(layer[i]) - 1):
            value = value + row[j] * layer[i][j + 1]
        netLayerList.append(findSigmoid(value))
    return netLayerList

# Forward Pass
def forwardPass(network, row):
    networkNetList = [row]
    # input for the next layer
    netLayerList = row
    for layer in network:
        netLayerList = findNetList(layer, netLayerList)
        networkNetList.append(netLayerList)
    return networkNetList


# Calculate delta
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


# Backward Pass
def backwardPass(network, networkNetList, row):
    length = len(networkNetList)
    targetIndex = len(row)-1
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
        deltaList = findDelta(layer, networkNetList[length-1], deltaList)
        del networkNetList[-1]
    return


# Calculate mean squared error
def findError(output,target):
    error = 0.5 * math.pow((target-output), 2)
    return error


def printNetworkWeights(network):
    print("-------------------------------------------------------------")
    print("Note: First term of every neuron is bias")
    print("-------------------------------------------------------------")
    layerCount = 0;
    for layer in network:
        layerCount = layerCount+1
        print('Layer %s : ' %(layerCount))
        for i in range(len(layer)):
            print("     Neuron[%s] weight: %s" %(i+1,layer[i]))
    print("-------------------------------------------------------------")
def findTestError(test, network):
    totalTestError = 0
    for index, row in test.iterrows():
        # forward pass
        networkNetList = forwardPass(network, row[:-1])
        n = len(networkNetList)
        finalOutputList = networkNetList[n - 1]
        target = row[(len(row) - 1)]
        totalTestError = totalTestError + findError(finalOutputList[0], target)

    meanTestError = totalTestError / len(test)
    return meanTestError

def main():

    training_percent = float(sys.argv[2]) if len(sys.argv) > 1 else 0.8
    maxiterations = int(sys.argv[3]) if len(sys.argv) > 1 else 20
    numberofhiddenlayers = int(sys.argv[4]) if len(sys.argv) > 1 else 2

    hiddenneuronsList = []
    if len(sys.argv) > 1:
        for i in range(numberofhiddenlayers):
            hiddenneuronsList.append(int(sys.argv[5+i]))
    else:
        hiddenneuronsList = [4]


    outputNeurons = 1

    # Dividing data into training and test randomly
    df = pd.DataFrame(np.random.randn(100, 2))
    msk = np.random.rand(len(input_dataset)) < training_percent
    train = input_dataset[msk]
    print("Training date size: ", len(train))
    test = input_dataset[~msk]
    print("Test data size:     ", len(test))

    seed(1)
    numberOfInputs = len(train.columns) - 1   # All except target value
    # parameters: Number of inputs, List having number of neurons in each hidden layer, Number of outputs
    network = initialize_newnetwork(numberOfInputs, hiddenneuronsList, outputNeurons)
    error = -1
    while error != 0 and maxiterations != 0:
        for index, row in train.iterrows():
            # forward pass
            networkNetList = forwardPass(network, row[:-1])
            n = len(networkNetList)
            finalOutputList = networkNetList[n-1]

            # backward pass
            backwardPass(network, networkNetList, row)

            target = row[(len(row)-1)]
            error = findError(finalOutputList[0],target)
        # print("Error ", error)
        maxiterations = maxiterations - 1

    # Print network weights
    printNetworkWeights(network)

    print("Training Error: ", error)

    # Calculate mean test error
    meanTestError = findTestError(test, network)

    print("Test Error:     ", meanTestError)

main()