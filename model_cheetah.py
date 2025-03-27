# Homework 2
# Problem: 1 and 2
# Due: 2021.03.08
# Desc: Trains a multi-layer perceptron (MLP) for two different problems.
# Problem 1: learn the function f(x)=1/x.
# Problem 2: build a 3-class classifier that models the IRIS data set.
from math import exp
from math import sqrt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


# Sigmoid activation function
def sigmoid(v):
    return 1.0 / (1.0 + exp(-v))

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 0 if x <= 0 else 1


# Derivative of Sigmoid activation function
def sigmoid_derivative(v):
    return v * (1.0 - v)


# Object representing each neuron/node in each layer
# Maintains the weights, current output, and error delta for the given node.
class Neuron:
    def __init__(self, weights, output, error_delta):
        self.weights = weights
        self.output = output
        self.error_delta = error_delta

    def activate(self, x):
        a = np.dot(x, self.weights[:-1]) + self.weights[-1]
        self.output = sigmoid(a)
        return self.output


# Pass the input sample through each layer of the network and return the output
def forward(x, weights):
    layer_input = x
    for layer in weights:
        new_input = list()
        for neuron in layer:
            output = neuron.activate(layer_input)
            new_input.append(output)
        layer_input = np.asarray(new_input)
    return layer_input


# Propagate the error back through the network
def backward(y, weights):

    # Initialize error at output layer
    curr_layer = weights[-1]
    for curr_node in range(0, len(y)):
        e = y[curr_node] - curr_layer[curr_node].output
        curr_layer[curr_node].error_delta = e * sigmoid_derivative(curr_layer[curr_node].output)

    # Propagate error back through hidden layers
    for curr_layer in range(len(weights) - 2, -1, -1):
        for curr_node in range(0, len(weights[curr_layer])):
            e = 0.0
            node = weights[curr_layer][curr_node]
            for forward_node in range(0, len(weights[curr_layer + 1])):
                w = weights[curr_layer + 1][forward_node].weights[curr_node]
                ed = weights[curr_layer + 1][forward_node].error_delta
                e += w * ed
            node.error_delta = e * sigmoid_derivative(node.output)


# Updates the network weights
def update(x, eta, weights):
    inputs = x
    for curr_layer in range(0, len(weights)):
        layer = weights[curr_layer]
        for curr_node in range(0, len(layer)):
            node = layer[curr_node]
            for i in range(0, len(node.weights) - 1):
                node.weights[i] += eta * node.error_delta * inputs[i]
            node.weights[-1] += eta * node.error_delta
        inputs = np.zeros(len(layer))
        for i in range(0, len(layer)):
            inputs[i] = layer[i].output


# Train the network
# X: training data
# Y: target outputs for each training sample
# layout: array containing the number of nodes in each layer
#   The first element of the array is the number of inputs.
#   The second, third, etc. elements of the array are the number of nodess in each hiden layer.
#   The last element of the array is the number of desired outputs.
#   e.g. [#inputs, layer1 nodes, layer2 nodes, layerN nodes, #outputs]
# eta: learning rate.  If not specified, the default is eta=1/sqrt(#inputs)
# decay: the rate of decay, default 0.0
# stoperr: optional stopping condition, when MSE gets below threshold, it stops
# Returns: weights, msehist
#       weights: represent the trained network model
#       msehist: the history of training error over the entire training process
def train(X, Y, layout, epochs, eta=None, decay=0.0, stoperr=None):
    weights = list()
    outputs = list()

    # Default learning rate eta=1/sqrt(#inputs)
    if eta is None:
        eta = 1 / sqrt(len(X[0]))

    # Initialize network with random weights
    for curr_layer in range(1, len(layout)):
        layer = list()
        outputs.append(np.zeros(layout[curr_layer]))
        numinputs = layout[curr_layer - 1]
        for i in range(0, layout[curr_layer]):
            layer.append(Neuron(np.random.rand(numinputs + 1), 0, 0))
        weights.append(layer)

    # Loop for # of epochs
    lrate = eta
    msehist = list()
    for e in range(0, epochs):
        err = 0.0

        # Train on each sample
        iter = 0
        xorder = np.arange(len(X))
        np.random.shuffle(xorder)
        for i in xorder:
            x = X[i]
            y = Y[i]
            yhat = forward(x, weights)
            backward(y, weights)

            # Compute MSE
            cerr = 0.0
            for j in range(0, len(y)):
                cerr += (y[j] - yhat[j]) ** 2
            cerr /= len(y)
            iter += 1
            err += cerr

            # Update weights with online learning
            update(x, lrate, weights)
        mse = err / len(X)
        msehist.append([e, 0, mse])
        print("MSE: " + str(mse))

        # Stopping condition if MSE is very low
        if stoperr is not None and mse < stoperr:
            break
        lrate = eta * (1.0 / (1.0 + decay * e))

    return weights, msehist


#################### INITIALIZE ###################
seed = 1
np.random.seed(seed)

train1 = Image.open('img/spot6.jpeg')
train2 = Image.open('img/spot5.jpeg')
train3 = Image.open('img/spot6.jpeg')
train_neg1 = Image.open('img/train_neg1.jpeg')
train_neg2 = Image.open('img/train_neg2.jpeg')
train_neg3 = Image.open('img/train_neg3.jpeg')
test1 = Image.open('img/test1.jpg')
test1 = ImageOps.grayscale(test1)
train1 = ImageOps.grayscale(train1)
train2 = ImageOps.grayscale(train2)
train3 = ImageOps.grayscale(train3)
train_neg1 = ImageOps.grayscale(train_neg1)
train_neg2 = ImageOps.grayscale(train_neg2)
train_neg3 = ImageOps.grayscale(train_neg3)
Xtrain = list()
Ytrain = list()
Xtest = list()
Ytest = list()
#Xtrain.append(list(train1.getdata()))
#Ytrain.append([1])
#Xtrain.append(list(train2.getdata()))
#Ytrain.append([1])
#Xtrain.append(list(train3.getdata()))
#Ytrain.append([1])
#Xtrain.append(list(train_neg1.getdata()))
#Ytrain.append([0])
#Xtrain.append(list(train_neg2.getdata()))
#Ytrain.append([0])
#Xtrain.append(list(train_neg3.getdata()))
#Ytrain.append([0])

pcatrans = PCA(n_components=100)
for x in range(0, 10):#600):
    for y in range(0, 10):#300):
        img = test1.crop((x, y, x + 25, y + 25))
        img = ImageOps.grayscale(img)
        l = list(img.getdata())
        Xtrain.append([v / 255 for v in l])
        Ytrain.append([0])

for x in range(250, 250+10):#600):
    for y in range(125, 125+10):#175):
        img = test1.crop((x, y, x + 25, y + 25))
        img = ImageOps.grayscale(img)
        l = list(img.getdata())
        Xtrain.append([v / 255 for v in l])
        Ytrain.append([1])

##########
for x in range(0, 20):#600):
    for y in range(0, 20):#300):
        img = test1.crop((x, y, x + 25, y + 25))
        img = ImageOps.grayscale(img)
        l = list(img.getdata())
        Xtest.append([v / 255 for v in l])
        Ytest.append([0])

for x in range(250, 250+20):#600):
    for y in range(125, 125+20):#175):
        img = test1.crop((x, y, x + 25, y + 25))
        img = ImageOps.grayscale(img)
        l = list(img.getdata())
        Xtest.append([v / 255 for v in l])
        Ytest.append([1])

df = pd.DataFrame(Xtrain)
dftest = pd.DataFrame(Xtest)
df = pcatrans.fit_transform(df)
dftest = pcatrans.transform(dftest)

w, msehist = train(df, Ytrain, [100, 50, 1], 1000, decay=0.01, stoperr=0.01)

for i in range(0, len(Ytrain)):
    yhat = forward(df[i], w)
    print(str(Ytrain[i]) + ", " + str(yhat))



#################### PROBLEM 2 ###################

# Load & pre-process IRIS data; use one-hot encoding for output classes
X = list()
Y = list()
xmax = np.zeros(4)
iris = pd.read_csv('iris.data', header=None)
for i in range(0, len(iris)):
    x = iris.iloc[i, :4].to_numpy()
    for j in range(0, len(x)):
        xmax[j] = max(xmax[j], x[j])
    X.append(x)
    yd = iris.iloc[i, 4]
    y = np.zeros(3)
    if yd == 'Iris-setosa':
        y[0] = 1.0
    elif yd == 'Iris-versicolor':
        y[1] = 1.0
    elif yd == 'Iris-virginica':
        y[2] = 1.0
    Y.append(y)
for i in range(0, len(X)):
    for j in range(0, len(X[i])):
        X[i][j] /= xmax[j]

# K-Fold Cross Validation
numright = numwrong = 0
acchist = list()
folds = 3
k = 0
kf = StratifiedKFold(n_splits=folds, random_state=seed)
for trainIndex, testIndex in kf.split(iris.loc[:, :4], iris.loc[:, 4]):

    # Split into training & test sets
    k += 1
    trainX = [X[index] for index in trainIndex]
    trainY = [Y[index] for index in trainIndex]
    testX = [X[index] for index in testIndex]
    testY = [Y[index] for index in testIndex]

    # Train the network
    w, msehist = train(trainX, trainY, [4, 5, 2, 3], 1000, decay=0.01, stoperr=0.01)

    # Evaluate on the test set
    numright = numwrong = 0
    acc = 0.0
    for i in range(0, len(testX)):
        yhat = forward(testX[i], w)
        y = testY[i]

        e = 0.0
        for j in range(0, len(y)):
            e += (y[j] - yhat[j]) ** 2
        e = e / len(y)
        print(round(e,6))

        if np.argmax(y) == np.argmax(yhat):
            numright += 1
        else:
            numwrong += 1
    acc = numright / (numright + numwrong)
    acchist.append(acc)
    print("Acc: " + str(acc))

# Output overall accuracy
avgacc = np.sum(acchist) / folds
print("Avg Acc: " + str(avgacc))
