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
    X_train, y_train, X_test, y_test = utils.split_data(X, y)
    X_train, X_test = utils.normalize(X_train), utils.normalize(X_test)
    lr = LinearRegression()
    lr.fit(X=X_train, y=y_train)
    y_train_pred, y_test_pred = lr.predict(X_train), lr.predict(X_test)
    train_abs_err = np.sum(np.abs(y_train_pred - y_train))
    test_abs_err = np.sum(np.abs(y_test_pred - y_test))
    print "train sae: {} , test sae: {}".format(train_abs_err, test_abs_err)
