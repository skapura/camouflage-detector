# Final project
# Nicholas Skapura
# Camouflage detection and assessment using AUTOENCODERS
# Original code based on MLP implemented for Homework #2
from math import exp
from math import sqrt
import numpy as np
import pandas as pd
import random
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from itertools import chain


# Sigmoid activation function
def sigmoid(v):
#    return 1.0 / (1.0 + exp(-v))

    if v < 0:
        return 1 - 1 / (1 + exp(v))
    return 1 / (1 + exp(-v))

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


def mse(y, yhat):
    cerr = 0.0
    for j in range(0, len(y)):
        cerr += (y[j] - yhat[j]) ** 2
    cerr /= len(y)
    return cerr


# Blend natural scenery into camouflage
# Simulate partially obstructed test images
# with Guassian blending.
# 1. Selects a random nature sample
# 2. Select a % of random Guassian pixels from the image
# 3. Blend those selected pixels into the camouflage image
# x - Camo image
# nature - Set of natural images
# noise_amt - Percentage of noise pixels to sample from
#               the natural image.
# Returns: a blended camouflage image with natural scenery.
def blend_natural(x, nature, noise_amt):
    num_pixels = int(len(x) * noise_amt)
    nsample = nature[np.random.randint(len(nature))]
    indexes = np.random.choice(len(x), size=num_pixels, replace=False)
    for i in indexes:
        x[i] = nsample[i]
    return x


#################### INITIALIZE ###################
seed = 1
np.random.seed(seed)

X = list()
Y = list()

# Load test data
for i in range(0, 100):
    img = Image.open('img/train/nature_a_' + str(i) + '.jpg')
    l = list(img.getdata())
    l = list(chain.from_iterable(l))
    X.append([v / 255 for v in l])
    Y.append(0)

    img = Image.open('img/train/pencott_' + str(i) + '.jpg')
    l = list(img.getdata())
    l = list(chain.from_iterable(l))
    X.append([v / 255 for v in l])
    Y.append(1)

# Shuffle training data
idx = list(range(0, len(X)))
random.shuffle(idx)
xbuf = list()
ybuf = list()
for i in idx:
    xbuf.append(X[i])
    ybuf.append(Y[i])
X = xbuf
Y = ybuf

# Experiment setup
dfx = pd.DataFrame(X)
dfx[dfx.shape[1]] = Y
folds = 3
k = 0
acc = 0.0
f = 0.0
results = list()
kf = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)

# Test each fold
for trainIndex, testIndex in kf.split(dfx.loc[:, :dfx.shape[1]], dfx.loc[:, dfx.shape[1]-1]):

    # Split into training & test sets
    k += 1
    trainX = [X[index] for index in trainIndex]
    trainY = [[Y[index]] for index in trainIndex]
    testX = [X[index] for index in testIndex]
    testY = [[Y[index]] for index in testIndex]

    # Train auto-encoder only on camouflage samples
    trainpos = list()
    for i in range(0, len(trainX)):
        if trainY[i][0] == 1:
            trainpos.append(trainX[i])

    # Find # of dimensions at 95% variance (based on camo data only)
    pcatrans = PCA()
    pcatrans.fit(trainpos)
    cumsum = np.cumsum(pcatrans.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    pcatrans = PCA(n_components=d)

    # Reduce dimensionality
    trainpos = pcatrans.fit_transform(trainpos)
    testX = pcatrans.transform(testX)
    xdim = len(trainpos[0])

    # Separate natural training samples
    testneg = list()
    for i in range(0, len(testX)):
        if testY[i][0] == 0:
            testneg.append(testX[i])

    #w, msehist = train(trainpos, trainpos, [xdim, int(xdim / 2), xdim], 200, eta=3.0,
    #                   decay=0.01, stoperr=0.01)
    w, msehist = train(trainpos, trainpos, [xdim, int(xdim / 2), xdim], 400, eta=1.0,
                       decay=0.01, stoperr=0.01)

    # Estimate error range on training data
    avgpos = 0.0
    for i in range(0, len(trainpos)):
        yhat = forward(trainpos[i], w)
        e = mse(trainpos[i], yhat)
        avgpos += e
    avgpos /= len(trainpos)

    # Evaluate performance
    tp = fp = tn = fn = 0
    avgpos = avgneg = 0.0
    for i in range(0, len(testX)):
        testsample = blend_natural(testX[i], testneg, 0.3)
        yhat = forward(testsample, w)
        e = mse(testX[i], yhat)
        if testY[i][0] == 0:
            avgneg += e
        else:
            avgpos += e
        ypred = 0 if e >= 2.5 else 1
        #ypred = 0 if e >= 1.5 else 1
        if ypred == 0 and testY[i][0] == 0:
            tn += 1
        elif ypred == 0 and testY[i][0] == 1:
            fn += 1
        elif ypred == 1 and testY[i][0] == 0:
            fp += 1
        elif ypred == 1 and testY[i][0] == 1:
            tp += 1
        print(str(e) + '<->' + str(testY[i]))
    acc += (tp + tn) / (tp + tn + fp + fn)
    f += tp / (tp + 0.5 * (fp + fn))
    print('pos: ' + str(avgpos / len(testY)))
    print('neg: ' + str(avgneg / len(testY)))
    print('fold ' + str(k))
    print('TP: ' + str(tp))
    print('TN: ' + str(tn))
    print('FP: ' + str(fp))
    print('FN: ' + str(fn))
    print((tp + tn) / (tp + tn + fp + fn))
    results.append([k, tp, tn, fp, fn])
    print('fold')

print(acc / 3.0)
print(f / 3.0)
