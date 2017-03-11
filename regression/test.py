from linear_regression import LinearRegression
import sys; sys.path.append('../')
from utils import utils
from loss import loss
import numpy as np

if __name__ == '__main__':
    with open('../data/housing.txt') as f:
        j = f.readlines()

    data = [map(float, s.split()) for s in j]
    data = np.array(data)
    X, y = data[:,:-1], data[:,-1]
    print X.shape
    print y.shape
