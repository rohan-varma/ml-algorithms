"""Implementation of Linear Regression for binary classification."""
import sys
sys.path.append('../')
from utils import utils
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn import metrics

class LinearRegression(object):
    def __init__(self, dim = None):
        if dim:
            self.weights = np.random.random(dim)

    def normalize(X):
        return (X - np.mean(X, axis = 1)) /np.std(X, axis = 1)

    def grad_desc_fit(X, y, max_iter = 1000, alpha = 0.01):
        # normalize X
        if not self.weights or self.weights.shape[0] != X.shape[1]:
            self.weights = np.random.random(X.shape[1])
        for i in range(max_iter):
            h_theta = X.dot(self.weights)
            grad = (h_theta - y).T.dot(X)
            self.weights = self.weights - (alpha * grad)
        return self.weights


    def fit(X, y):
        if not self.weights or self.weights.shape[0] != X.shape[1]:
            self.weights = np.random.random(X.shape[1])
        self.weights = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        return self.weights

    def predict(X):
        return X.dot(self.weights)

    def cost(X, y):
        return np.dot((predict(X) - y).dot((predict(X) - y).T))


if __name__ == '__main__':
    with open('./data/housing.txt') as f:
        whole_data = f.readlines()
    data = np.array([map(lambda z: float(z), item.split())
                     for item in whole_data])

    X, y = data[:,:-1], data[:,-1] # split
    X = (X - np.mean(X, axis = 0))/np.std(X, axis = 0) # normalize
    X_train, X_test = X[:int(0.8 * X.shape[0])], X[int(0.8*X.shape[0]):]
    y_train, y_test = y[:int(0.8 * y.shape[0])], y[int(0.8*y.shape[0]):]
