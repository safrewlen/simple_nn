import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


class SimpleNN:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)  # creates nd array (num of columns * 4 )
        self.layer1 = np.random.rand(self.input.shape[0], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y  # y_true
        self.output = np.zeros(self.y.shape)

    def feed_forward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))  # np.dot(a, b) = a@b, matrix multiplication
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def back_propagation(self):
        self.loss = (self.output - self.y)**2
