import unittest
from logistic_regression import LogisticRegression
from linear_regression import LinearRegression
from perceptron import Perceptron
from softmax_regression import SoftmaxRegression
import numpy as np
import sys
sys.path.append('../')
from utils import utils
from sklearn import metrics

class LinearAlgorithmsTests(unittest.TestCase):
    """Tests for linear machine learning algorithms"""

    def setup(self):
        self.linear_regression = LinearRegression()
        self.logistic_regression = LogisticRegression()
        self.perceptron = Perceptron()
        sefl.softmax_regression = SoftmaxRegression()

    def test(self):
        self.assertTrue(True)



if __name__ == '__main__':
    unittest.main()
