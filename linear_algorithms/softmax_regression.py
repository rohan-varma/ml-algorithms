"""Implementation of softmax regression for multinomial classification"""
import sys
sys.path.append('../')
from utils import utils
import numpy as np
import sklearn.linear_model
from sklearn import metrics

class SoftmaxRegression(object):

    def __init__(self, in_dim=None, out_dim=None):
        if in_dim and out_dim:
            self.weights = np.random.random((out_dim, in_dim))

    def fit(self, X, y):
        pass

    def onehot_encode(y):
        pass

    def cost(X, y):
        pass

    def softmax(vec):
        return np.exp(vec)/np.sum(np.exp(vec), axis = 0) # TODO this is not a great implementation of this...

    def predict(X):
        lin_comb = self.weights.dot(X.T)
        return softmax(lin_comb)


if __name__ == "__main__":
    softax = SoftmaxRegression()
