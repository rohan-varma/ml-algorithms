"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
"""

# This code was adapted from course material by Jenna Wiens (UMichigan).

# python libraries
import os

# numpy libraries
import numpy as np
import time
# matplotlib libraries
import matplotlib.pyplot as plt
######################################################################
# classes
######################################################################

class Data :

    def __init__(self, X=None, y=None) :
        """
        Data class.

        Attributes
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        """

        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y

    def load(self, filename) :
        """
        Load csv file into X array of features and y array of labels.

        Parameters
        --------------------
            filename -- string, filename
        """

        # determine filename
        dir = os.path.dirname(__file__)
        f = os.path.join(dir, '..', 'data', filename)

        # load data
        with open(f, 'r') as fid :
            data = np.loadtxt(fid, delimiter=",")

        # separate features and labels
        self.X = data[:,:-1]
        self.y = data[:,-1]

    def plot(self, **kwargs) :
        """Plot data."""

        if 'color' not in kwargs :
            kwargs['color'] = 'b'

        plt.scatter(self.X, self.y, **kwargs)
        plt.xlabel('x', fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        plt.show()

# wrapper functions around Data class
def load_data(filename) :
    data = Data()
    data.load(filename)
    return data

def plot_data(X, y, **kwargs) :
    data = Data(X, y)
    data.plot(**kwargs)


class PolynomialRegression() :

    def __init__(self, m=1, reg_param=0) :
        """
        Ordinary least squares regression.

        Attributes
        --------------------
            coef_   -- numpy array of shape (d,)
                       estimated coefficients for the linear regression problem
            m_      -- integer
                       order for polynomial regression
            lambda_ -- float
                       regularization parameter
        """
        self.coef_ = None
        self.m_ = m
        self.lambda_ = reg_param


    def generate_polynomial_features(self, X) :
        """
        Maps X to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].

        Parameters
        --------------------
            X       -- numpy array of shape (n,1), features

        Returns
        --------------------
            Phi     -- numpy array of shape (n,(m+1)), mapped features
        """

        n,d = X.shape
        # print X.shape
        # exit()

        ### ========== TODO : START ========== ###
        # part b: modify to create matrix for simple linear model
        # part g: modify to create matrix for polynomial model
        Phi = X
        m = self.m_
        Phi = np.array([map(lambda n: pow(x[0], n), range(0,m+1)) for x in X])





        ### ========== TODO : END ========== ###

        return Phi


    def fit_GD(self, X, y, eta=None,
                eps=0, tmax=10000, verbose=False) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            eta     -- float, step size
            eps     -- float, convergence criterion
            tmax    -- integer, maximum number of iterations
            verbose -- boolean, for debugging purposes

        Returns
        --------------------
            self    -- an instance of self
        """
        t0 = time.clock()
        if self.lambda_ != 0 :
            raise Exception("GD with regularization not implemented")

        if verbose :
            plt.subplot(1, 2, 2)
            plt.xlabel('iteration')
            plt.ylabel(r'$J(\theta)$')
            plt.ion()
            plt.show()

        X = self.generate_polynomial_features(X) # map features
        n,d = X.shape
        eta_input = eta
        self.coef_ = np.zeros(d)                 # coefficients
        err_list  = np.zeros((tmax,1))           # errors per iteration

        # GD loop
        for t in xrange(tmax) :
            ### ========== TODO : START ========== ###
            # part f: update step size
            # change the default eta in the function signature to 'eta=None'
            # and update the line below to your learning rate function
            if eta_input is None :

                eta = 1/float(1+t+1) # change this line
            else :
                eta = eta_input
            if t == 0:
                print "fitting with eta: " + str(eta)
            ### ========== TODO : END ========== ###

            ### ========== TODO : START ========== ###
            # part d: update theta (self.coef_) using one step of GD
            # hint: you can write simultaneously update all theta using vector math
            self.coef_ = self.coef_ - eta*(np.dot((np.dot(X.T, X)), self.coef_) - np.dot(X.T, y))
            # track error
            # hint: you cannot use self.predict(...) to make the predictions
            y_pred = np.dot(self.coef_, X.T)
            err_list[t] = np.sum(np.power(y - y_pred, 2)) / float(n)
            ### ========== TODO : END ========== ###

            # stop? Fixed bug ( changed the termination condition from < eps to <= eps)
            if t > 0 and abs(err_list[t] - err_list[t-1]) <= eps :
                break

            # debugging
            if verbose :
                x = np.reshape(X[:,1], (n,1))
                cost = self.cost(x,y)
                plt.subplot(1, 2, 1)
                plt.cla()
                plot_data(x, y)
                self.plot_regression()
                plt.subplot(1, 2, 2)
                plt.plot([t+1], [cost], 'bo')
                plt.suptitle('iteration: %d, cost: %f' % (t+1, cost))
                plt.draw()
                plt.pause(0.05) # pause for 0.05 sec

        print 'number of iterations: %d' % (t+1)
        num_iters = t + 1
        return self, num_iters, time.clock() - t0


    def fit(self, X, y, l2regularize = None ) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using the closed form solution.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            l2regularize    -- set to None for no regularization. set to positive double for L2 regularization

        Returns
        --------------------
            self    -- an instance of self
        """
        t0 = time.clock()
        X = self.generate_polynomial_features(X) # map features

        ### ========== TODO : START ========== ###
        # part e: implement closed-form solution
        # hint: use np.dot(...) and np.linalg.pinv(...)
        #       be sure to update self.coef_ with your solution
        self.coef_ = ((np.linalg.pinv(np.dot(X.T,X))).dot(X.T)).dot(y)
        return time.clock() - t0

        ### ========== TODO : END ========== ###


    def predict(self, X) :
        """
        Predict output for X.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features

        Returns
        --------------------
            y       -- numpy array of shape (n,), predictions
        """
        if self.coef_ is None :
            raise Exception("Model not initialized. Perform a fit first.")

        X = self.generate_polynomial_features(X) # map features

        ### ========== TODO : START ========== ###
        # part c: predict y
        # print self.coef_
        y = np.dot(self.coef_, X.T)
        ### ========== TODO : END ========== ###

        return y


    def cost(self, X, y) :
        """
        Calculates the objective function.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets

        Returns
        --------------------
            cost    -- float, objective J(theta)
        """
        ### ========== TODO : START ========== ###
        # part d: compute J(theta)
        cost = sum([(self.predict(X=X) - y)**2])


        ### ========== TODO : END ========== ###
        return sum(cost)


    def rms_error(self, X, y) :
        """
        Calculates the root mean square error.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets

        Returns
        --------------------
            error   -- float, RMSE
        """
        ### ========== TODO : START ========== ###
        # part h: compute RMSE
        error = np.sqrt(sum(sum([(self.predict(X=X) - y)**2]))/float(X.shape[0]))
        ### ========== TODO : END ========== ###
        return error


    def plot_regression(self, xmin=0, xmax=1, n=50, **kwargs) :
        """Plot regression line."""
        if 'color' not in kwargs :
            kwargs['color'] = 'r'
        if 'linestyle' not in kwargs :
            kwargs['linestyle'] = '-'

        X = np.reshape(np.linspace(0,1,n), (n,1))
        y = self.predict(X)
        plot_data(X, y, **kwargs)
        plt.show()


######################################################################
# main
######################################################################

def main() :
    # load data
    train_data = load_data('regression_train.csv')
    test_data = load_data('regression_test.csv')



    ### ========== TODO : START ========== ###
    # part a: main code for visualizations
    print 'Visualizing data...'
    X_train, y_train = train_data.X, train_data.y
    # plot_data(X_train, y_train)
    # exit()


    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # parts b-f: main code for linear regression
    print 'Investigating linear regression...'
    train_data = load_data('regression_train.csv')
    model = PolynomialRegression()
    model.coef_ = np.zeros(2)
    print model.cost(train_data.X, train_data.y)
    print "FITTING WITH GD"
    etas = [0.0001, 0.001, 0.01, 0.0407]
    li = []
    for eta in etas:
        model, num_iters, time= model.fit_GD(X=train_data.X, y=train_data.y, eta = eta, verbose = False)
        li.append({'eta': eta, 'coefficient': model.coef_, 'num_iters': num_iters, 'cost': model.cost(train_data.X,train_data.y), 'time': time})


    for item in li:
        print "eta: " + str(item['eta']) + " | " + "coefficient: " + str(item['coefficient']) + " | " + "cost: " + str(item['cost']) + " | " + 'num_iterations: ' + str(item['num_iters']) + " | " + 'time: ' + str(item['time'])
    print "Fitting with closed form solution"
    tclosed = model.fit(X=train_data.X, y = train_data.y)
    print model.cost(train_data.X, train_data.y)
    print tclosed
    print "closed for coefs: " + str(model.coef_)
    print "fitting with decaying LR"
    model, num_iters, tm=model.fit_GD(X=train_data.X, y=train_data.y, eta = None, verbose = False)
    print "decay:"
    print "num_iters: " + str(num_iters) + " time: " + str(tm) + " cost: " + str(model.cost(train_data.X, train_data.y))
    print "decay coefs: " + str(model.coef_)


    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # parts g-i: main code for polynomial regression
    print 'Investigating polynomial regression...'
    nomials = range(0,11)
    d_train = {}
    d_test = {}
    for m in nomials:
        model = PolynomialRegression(m=m)
        model.fit(X=train_data.X, y = train_data.y)
        d_train[m]=model.rms_error(train_data.X, train_data.y)
        d_test[m]=model.rms_error(test_data.X, test_data.y)
    plt.plot(d_train.keys(), d_train.values(), 'b--', label='training err')
    plt.plot(d_test.keys(), d_test.values(), 'g--', label = 'test err')
    plt.xlabel('model complexity')
    plt.ylabel('root mse')
    plt.show()



    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # parts j-k (extra credit): main code for regularized regression
    print 'Investigating regularized regression...'

    ### ========== TODO : END ========== ###



    print "Done!"

if __name__ == "__main__" :
    main()
