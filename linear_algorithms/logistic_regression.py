"""Implementation of Logistic Regression for binary classification."""
import sys
sys.path.append('../')
from utils import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

class LogisticRegression(object):
    def __init__(self, dim=None):
        if dim:
            self.weights = np.random.uniform(-1, 1, dim)
        else:
            self.weights = None

    def fit(self, X, y, max_iters = 10000, eps = 0.0001, alpha = 0.01,
            decay_rate = True, verbose = False):
        # initialize weights
        if not self.weights or self.weights.shape[0] != X.shape[1]:
            self.weights = np.random.uniform(-1, 1, X.shape[1])

        for i in max_iters:
            y_hat = self.predict(X)
            grad = X * (y_hat - y)
            self.weights+= -alpha * grad
            if verbose and (i+1) % 100 ==0:
                print "iteration: " + str(i + 1)
                print "cost: " + str(self.cost(y_hat,y))

        return self


    def predict(self, X):
        return 1 if sigmoid(X.dot(self.weights)) > 0.5 else 0

    def sigmoid(z):
        return expit(z)


    def cost(self, y_pred, y):
        # use the average negative log likelihood as the cost.
        n_ll = -1 * sum(map(lambda label: label * np.log(self.predict(X))
                       + (1 - label) * np.log(1 - self.predict(X)), y))
        return n_ll/float(y.shape[0])


if __name__ == '__main__':
    with open('./data/pima-diabetes.txt') as f:
        whole_data = f.readlines()
    data = np.array([item.split(",") for item in whole_data])
    train_data, test_data = data[:int(0.8*data.shape[0])], \
    data[int(0.8*data.shape[0]):]
