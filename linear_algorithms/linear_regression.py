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
        else:
            self.weights = None

    def normalize(self, X):
        return (X - np.mean(X, axis = 0)) /np.std(X, axis = 0)

    def grad_desc_fit(self, X, y, max_iter = 100000, alpha = 0.001, eps = 0.00001, verbose = True):
        # make sure that x is normalized enough
        mean, std = np.mean(X, axis = 0), np.std(X, axis = 0)
        if mean.any() < -1 or mean.any() > 1 or std.any() > 2:
            raise AssertionError("remember to normalize inputs!")
        if not self.weights or self.weights.shape[0] != X.shape[1]:
            self.weights = np.random.random(X.shape[1])
        if verbose:
            print "initial cost: " + str(self.cost(X,y))
        prev_cost = self.cost(X,y)
        for i in range(max_iter):
            X_T_X = X.T.dot(X)
            t2 = X_T_X.dot(self.weights)
            t3 = X.T.dot(y)

            grad = t2 - t3
            # print grad
            self.weights+= -(alpha * grad)
            new_cost = self.cost(X,y)
            # if np.abs(new_cost - prev_cost) < eps:
            #     break
            # else:
            #     prev_cost = new_cost
        if verbose:
            print "final cost, gradient descent: " + str(self.cost(X,y))
        return self.weights


    def fit(self, X, y, verbose = True):
        if not self.weights or self.weights.shape[0] != X.shape[1]:
            self.weights = np.random.random(X.shape[1])
        print "initial cost: " + str(self.cost(X,y))
        self.weights = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        print "final cost, explicit solution: " + str(self.cost(X,y))
        print self.weights
        return self.weights

    def predict(self, X):
        return X.dot(self.weights)

    def cost(self, X, y):
        return np.square(np.linalg.norm(self.predict(X)-y))/X.shape[0]


if __name__ == '__main__':
    with open('./data/housing.txt') as f:
        whole_data = f.readlines()
    data = np.array([map(lambda z: float(z), item.split())
                     for item in whole_data])

    X, y = data[:,:-1], data[:,-1] # split
    X = (X - np.mean(X, axis = 0))/np.std(X, axis = 0) # normalize
    X_train, X_test = X[:int(0.8 * X.shape[0])], X[int(0.8*X.shape[0]):]
    y_train, y_test = y[:int(0.8 * y.shape[0])], y[int(0.8*y.shape[0]):]
    lr = LinearRegression()
    lr.grad_desc_fit(X_train, y_train)
