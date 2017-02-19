"""Implementation of Logistic Regression for binary classification."""
import sys
sys.path.append('../')
from utils import utils
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn import metrics

class LogisticRegression(object):
    def __init__(self, dim = None):
        if dim:
            self.weights = np.random.random(dim)
        else:
            self.weights = None

    def sigmoid(self, z):
        return 1.0/(1 + np.exp(-z))

    def cost(self, X, y, reg_param = None):
        h_theta = self.sigmoid(X.dot(self.weights))
        nll =  -np.mean(y*np.log(h_theta) + (1-y)*np.log(1-h_theta))
        return nll if not reg_param else nll + reg_param*0.5*np.inner(
            self.weights,self.weights)

    def fit(self, X, y, alpha = 0.001, eps = 0.00001, max_iters = 200,
            minibatch = True, decay_rate = 0.00001, regularize_lambda = 0.5,
            early_termination = True, verbose = True):
        cost_iter = []
        cost = self.cost(X,y, regularize_lambda)
        cost_iter.append(cost)
        for i in range(max_iters):
            if decay_rate: alpha /= (1.0/(1 + decay_rate * i))
            prev_cost = cost_iter[-1]
            # gradient of the objective function
            grad = (self.sigmoid(X.dot(self.weights)) -y).T.dot(X)
            self.weights+=-alpha*(grad + regularize_lambda*self.weights) # add regularization param
            cost = self.cost(X,y, regularize_lambda)
            if np.abs(prev_cost - cost) < eps: break
            cost_iter.append(cost)
            if verbose:
                print "epoch: " + str(i + 1)
                print "cost: " + str(self.cost(X,y, regularize_lambda))

        return self.weights, np.array(cost_iter)

    def predict(self, X):
        probs = self.sigmoid(X.dot(self.weights))
        predictions = np.where(probs > 0.5,1,0)
        return predictions


def get_sklearn_scores(X, y):
    sklearn_logreg = sklearn.linear_model.LogisticRegression()
    sklearn_logreg.fit(X,y)
    y_pred = sklearn_logreg.predict(X)
    return 1 - metrics.accuracy_score(y_true=y, y_pred=y_pred, normalize=True)

if __name__ == '__main__':
    # get the data
    with open('./data/pima-diabetes.txt') as f:
        whole_data = f.readlines()
    data = np.array([map(lambda z: float(z), item.split(","))
                     for item in whole_data])

    # split and norm data
    train_data, test_data = data[:int(0.8*data.shape[0])], \
    data[int(0.8*data.shape[0]):]
    X_train, y_train = train_data[:,:-1], train_data[:,-1]
    X_test, y_test = test_data[:,:-1], test_data[:,-1]
    X_train = (X_train - np.mean(X_train, axis = 0)) / np.std(X_train, axis = 0)
    X_test = (X_test - np.mean(X_test, axis = 0)) / np.std(X_test, axis = 0)

    # logistic regression
    logreg = LogisticRegression(X_train.shape[1])
    print "initial error: " + str(1 - metrics.accuracy_score(y_train,logreg.predict(X_train)))
    weights, cost = logreg.fit(X_train,y_train,verbose=False)
    logreg.predict(X_train)
    print "final error: " + str(1 - metrics.accuracy_score(y_train,logreg.predict(X_train)))
    # plt.plot(range(len(cost)), cost)
    # plt.show()

    # using sklearn
    print get_sklearn_scores(X_train, y_train)

    # testing dataset
    print "test error: " + str(1 - metrics.accuracy_score(y_true=y_test, y_pred=logreg.predict(X_test)))
    print "sklearn test error: " + str(get_sklearn_scores(X_test, y_test))

    mean_train_err, mean_test_err = utils.get_errors_already_split(logreg, X_train, y_train, X_test, y_test)
    print "mean train error: " + str(mean_train_err)
    print "mean test error: " + str(mean_test_err)

    X = np.concatenate((X_train, X_test), axis = 0)
    y = np.concatenate((y_train, y_test), axis = 0)
    cv_train_err, cv_test_err = utils.cross_validate(logreg,X,y)
    print "cv train error: " + str(cv_train_err)
    print "cv test error: " + str(cv_test_err)
