"""Implementation of vanilla Linear Regression for binary classification."""
import sys
sys.path.append('../')
from utils import utils
from loss import loss
import numpy as np
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
        """Return the cost function, which is the squared L2-error"""
        return loss.squared_l2_loss(y=y, pred=self.predict(X), average=True)
