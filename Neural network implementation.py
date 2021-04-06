import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


def layer_sizes():
    n_x = 2
    n_h = 4
    n_y = 1
    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))

    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward_propagation(x, params):
    z1 = np.dot(params['w1'], x) + params['b1']
    a1 = relu(z1)

    z2 = np.dot(params['w2'], a1) + params['b2']
    a2 = sigmoid(z2)
    return {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}


def compute_cost(a2, y, params):
    logprobs = np.multiply(np.log(a2), y) + np.multiply(np.log(1 - a2), 1 - y)
    cost = np.sum(logprobs) / y.shape[1]
    return np.squeeze(cost)


def backward_propagation(parameters, cache, x, y):
    m = x.shape[1]
    dz2 = cache['a2'] - y
    dw2 = np.dot(dz2, cache['a1'].T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m

    dz1 = np.dot(parameters['w2'].T, dz2) * relu_derivative(cache['z1'])
    dw1 = np.dot(dz1, x.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m

    gradients = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}
    return gradients


def update_parameters(parameters, dv, learning_rate=1.2):
    parameters['w1'] = parameters['w1'] - learning_rate * dv['dw1']
    parameters['b1'] = parameters['b1'] - learning_rate * dv['db1']
    parameters['w2'] = parameters['w2'] - learning_rate * dv['dw2']
    parameters['b2'] = parameters['b2'] - learning_rate * dv['db2']
    return parameters


def nn_model(x, y, num_iterations=50):
    n_x = layer_sizes()[0]
    n_h = layer_sizes()[1]
    n_y = layer_sizes()[2]

    x = x.T
    y = y.T
    parameters = initialize_parameters(n_x, n_h, n_y)
    for i in range(0, num_iterations):
        results = forward_propagation(x, parameters)
        cost = compute_cost(results['a2'], y, parameters)
        print("epoch=", i+1, "    ", "cost=", cost)
        grads = backward_propagation(parameters, results, x, y)
        parameters = update_parameters(parameters, grads)

    return parameters


def predict(parameters, x):
    x = x.T
    results = forward_propagation(x, parameters)
    predictions = np.around(results['a2'])
    return predictions


'''Banknote Dataset : The Banknote Dataset involves predicting whether a given banknote is authentic given a number of measures taken from a photograph.
    Variance of Wavelet Transformed image (continuous) (used)
    Skewness of Wavelet Transformed image (continuous) (used)
    Kurtosis of Wavelet Transformed image (continuous)
    Entropy of image (continuous)
    Class (0 for authentic, 1 for inauthentic)'''

# only used first two attributes
dataset = pd.read_csv('data_banknote_authentication.txt', sep=",", header=None)
dataset.drop(dataset.columns[[2]], axis=1,inplace=True)
dataset.drop(dataset.columns[[2]], axis=1, inplace=True)
y = dataset.iloc[:, 2].to_frame()
y = y.values.reshape(y.shape[0], 1)
x = dataset.drop(dataset.columns[[2]], axis=1)

x = normalize(x)
y = normalize(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print("shape of original dataset :", dataset.shape)
print("shape of input - training set", x_train.shape)
print("shape of output - training set", y_train.shape)
print("shape of input - testing set", x_test.shape)
print("shape of output - testing set", y_test.shape)

parameters = nn_model(x_train, y_train, num_iterations=500)
predictions = predict(parameters, x_test)

print('accuracy=', float(np.dot(y_test.T, predictions.T) + np.dot(1 - y_test.T, 1 - predictions.T)) / float(y_test.size) * 100, '%')
