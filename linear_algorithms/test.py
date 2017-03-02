import unittest
from logistic_regression import LogisticRegression
from linear_regression import LinearRegression
from perceptron import Perceptron
from softmax_regression import SoftmaxRegression
import numpy as np
import sys
sys.path.append('../')
from utils import utils
from loss import loss
from sklearn import metrics

class LinearAlgorithmsTests(unittest.TestCase):
    """Tests for linear machine learning algorithms"""

    def test_loss_functions(self):
        # squared error loss - for linear regression
        a, b = np.arange(10), np.arange(10)
        self.assertEqual(loss.squared_l2_loss(a, b), 0)
        c, d = np.array((1,2)), np.array((5,8))
        self.assertAlmostEqual(loss.squared_l2_loss(c, d, average=False), 52)
        # testing zero/one loss
        y, p = np.array([0,1,0,0,1,1]), np.array([1,0,0,0,0,1])
        print np.sum(np.abs(y - p))/float(y.shape[0])
        self.assertAlmostEqual(loss.zero_one_loss(y, p, average=True), 3/float(y.shape[0]))



    def test_linear_regression(self):
        with open('./data/housing.txt') as f:
            whole_data = f.readlines()
        data = np.array([map(lambda z: float(z), item.split())
                         for item in whole_data])
        X, y = data[:,:-1], data[:,-1] # split
        X = (X - np.mean(X, axis = 0))/np.std(X, axis = 0) # normalize
        X_train, X_test = X[:int(0.8 * X.shape[0])], X[int(0.8*X.shape[0]):]
        y_train, y_test = y[:int(0.8 * y.shape[0])], y[int(0.8*y.shape[0]):]
        linear_regression = LinearRegression()
        linear_regression.fit(X_train, y_train)
        y_test_pred = linear_regression.predict(X_test)
        print "l2 error: " + str(np.linalg.norm(y_test_pred - y_test))
        return True




if __name__ == '__main__':
    unittest.main()
