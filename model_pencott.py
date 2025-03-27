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
import random
from PIL import Image, ImageOps, ImageDraw
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from itertools import chain


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
        print("Epoch: " + str(e) + ", MSE: " + str(mse))

        # Stopping condition if MSE is very low
        if stoperr is not None and mse < stoperr:
            break
        lrate = eta * (1.0 / (1.0 + decay * e))

    return weights, msehist


#################### INITIALIZE ###################
seed = 1
np.random.seed(seed)

train1 = Image.open('img/pencott_train_saturation.jpg')
test1 = Image.open('img/pencott_test2_saturation.jpg')
w, h = train1.size
Xtrain = list()
Ytrain = list()
Xtest = list()
Ytest = list()


# Negative class (natural scenery)
#for x in range(0, 200, 50):
#    for y in range(0, 1200, 50):
i = 0
for x in range(0, 100):
    for y in range(0, 100):
        img = train1.crop((x, y, x + 50, y + 50))
        img.save('img/train/nature_a_' + str(i) + '.jpg')
        i += 1
        #img = ImageOps.grayscale(img)
        l = list(img.getdata())
        l = list(chain.from_iterable(l))
        Xtrain.append([v / 255 for v in l])
        Ytrain.append([0])
#for y in range(0, h - 100, 20):
#    img = train1.crop((w - 100, y, w - 50, y + 50))
#    #img = ImageOps.grayscale(img)
#    l = list(img.getdata())
#    l = list(chain.from_iterable(l))
#    Xtrain.append([v / 255 for v in l])
#    Ytrain.append([0])

# Positive class (camouflaged object)
c = 0
for x in range(380, 380+100):
    for y in range(150, 150+100):

        img = train1.crop((x, y, x + 50, y + 50))
        img.save('img/train/pencott_' + str(c) + '.jpg')
        c += 1
        #img = ImageOps.grayscale(img)
        l = list(img.getdata())
        l = list(chain.from_iterable(l))
        Xtrain.append([v / 255 for v in l])
        Ytrain.append([1])

# Shuffle training data
idx = list(range(0, len(Xtrain)))
random.shuffle(idx)
xbuf = list()
ybuf = list()
for i in idx:
    xbuf.append(Xtrain[i])
    ybuf.append(Ytrain[i])
Xtrain = xbuf
Ytrain = ybuf

##########
testcoords = list()
fulltest = list()
for x in range(0, w - 50, 50):
    for y in range(0, h - 50, 50):
        img = test1.crop((x, y, x + 50, y + 50))
        #img = ImageOps.grayscale(img)
        l = list(img.getdata())
        l = list(chain.from_iterable(l))
        fulltest.append([v / 255 for v in l])
        testcoords.append((x, y))

for x in range(0, 20):
    for y in range(0, 20):
        img = train1.crop((x, y, x + 50, y + 50))
        #img = ImageOps.grayscale(img)
        l = list(img.getdata())
        l = list(chain.from_iterable(l))
        Xtest.append([v / 255 for v in l])
        Ytest.append([0])
        #testcoords.append((x, y))

for x in range(380, 380+20):
    for y in range(150, 150+20):
        img = train1.crop((x, y, x + 50, y + 50))
        #img = ImageOps.grayscale(img)
        l = list(img.getdata())
        l = list(chain.from_iterable(l))
        Xtest.append([v / 255 for v in l])
        Ytest.append([1])
        #testcoords.append((x, y))


dftrain = pd.DataFrame(Xtrain)
dftest = pd.DataFrame(Xtest)
dffull = pd.DataFrame(fulltest)

# Find # of dimensions at 95% variance
pcatrans = PCA()
pcatrans.fit(dftrain)
cumsum = np.cumsum(pcatrans.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
pcatrans = PCA(n_components=d)

# Reduce dimensionality
dftrain = pcatrans.fit_transform(dftrain)
dftest = pcatrans.transform(dftest)
dffull = pcatrans.transform(dffull)

w, msehist = train(dftrain, Ytrain, [dftrain.shape[1], dftrain.shape[1]*3, 10, 1], 100, eta=3.0, decay=0.01, stoperr=0.01)

numright = numwrong = 0
for i in range(0, len(Ytrain)):
    yhat = forward(dftrain[i], w)
    ypred = 0 if yhat < 0.5 else 1
    if ypred == Ytrain[i][0]:
        numright += 1
    else:
        numwrong += 1
    #print(str(Ytrain[i][0]) + ":" + str(ypred) + ", " + str(yhat))
print(numright)
print(numwrong)


numright = numwrong = 0
#imgout = Image.open('img/cadpat_marpat.jpg')
#dr = ImageDraw.Draw(imgout)
for i in range(0, len(Ytest)):
    yhat = forward(dftest[i], w)
    ypred = 0 if yhat < 0.5 else 1
    #if ypred == 1:
    #    dr.point(testcoords[i], fill="red")
    if ypred == Ytest[i][0]:
        numright += 1
    else:
        numwrong += 1
    print(str(Ytest[i][0]) + ":" + str(ypred) + ", " + str(yhat))
#imgout.save('img/cadpat_marpat_output.jpg')
print(numright)
print(numwrong)

imgout = Image.open('img/pencott_test2.jpg')
dr = ImageDraw.Draw(imgout, 'RGBA')
for i in range(0, len(dffull)):
    yhat = forward(dffull[i], w)
    print(yhat)
    ypred = 0 if yhat < 0.6 else 1
    if ypred == 1:
        print(testcoords[i])
        dr.rectangle((testcoords[i], (testcoords[i][0]+50, testcoords[i][1]+50)), fill=(255, 0, 0, 128))
imgout.save('img/pencott_test2_output.jpg')

print('done')