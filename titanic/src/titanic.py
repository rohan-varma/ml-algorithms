"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self

    def predict(self, X) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")

        n,d = X.shape
        y = [self.prediction_] * n
        return y


class RandomClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None

    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        n_zeros = len(filter(lambda l: l == 0, y))
        n_ones = len(filter(lambda l: l == 1, y))
        self.probabilities_ = {0: n_zeros/float(y.shape[0]), 1: n_ones/float(y.shape[0])}
        # print self.probabilities_
        #print sum(self.probabilities_.values())

        ### ========== TODO : END ========== ###

        return self

    def predict(self, X, seed=1234) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)


        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        # print self.probabilities_
        y = np.random.choice(2,X.shape[0],
                             p = [self.probabilities_[0],
                                  self.probabilities_[1]])

        ### ========== TODO : END ========== ###

        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in xrange(d) :
        fig.add_subplot (3,3,i)
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')

    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in xrange(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = range(int(math.floor(min(features))), int(math.ceil(max(features)))+1)
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'

    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)

    train_error = 0
    test_error = 0
    for trial in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=ntrials)
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        train_error += 1 - metrics.accuracy_score(y_train, y_pred_train, normalize=True)
        test_error += 1 - metrics.accuracy_score(y_test, y_pred_test, normalize=True)

    train_error = train_error/float(ntrials)
    test_error = test_error/float(ntrials)





    ### ========== TODO : END ========== ###

    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(zip(y_pred))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features



    #========================================
    # part a: plot histograms of each feature
    print 'Plotting...'
    # for i in xrange(d) :
    #     plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)


    #========================================
    # train Majority Vote classifier on data
    print 'Classifying using Majority Vote...'
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print '\t-- training error: %.3f' % train_error



    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print 'Classifying using Random...'
    randclass = RandomClassifier()
    randclass.fit(X, y)
    y_pred = randclass.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize = True)
    print '\t-- training error: %.3f' % train_error


    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain
    print 'Classifying using Decision Tree...'
    dclf = DecisionTreeClassifier(criterion="entropy")
    dclf.fit(X, y)
    y_pred = dclf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print '\t-- training error: %.3f' % train_error



    ### ========== TODO : START ========== ###
    # part d: use cross-validation to compute average training and test error of classifiers
    print 'Investigating various classifiers...'
    train_err_rand, test_err_rand = error(randclass, X, y)
    train_err_maj, test_err_maj = error(clf, X, y)
    train_err_decision, test_err_decision = error(dclf, X, y)
    print "randclass training error: " + str(train_err_rand)
    print "randclass testing error: " + str(test_err_rand)
    print "maj class training error: " + str(train_err_maj)
    print "maj class test error: " + str(test_err_maj)
    print "dtree training error: " + str(train_err_decision)
    print "dtree testing error: " + str(test_err_decision)


    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part e: investigate decision tree classifier with various depths
    print 'Investigating depths...'
    depths = np.arange(1,21)
    training_errors, testing_errors = [], []
    for depth in depths:
        # instantiate a classifier
        dclf = DecisionTreeClassifier(criterion="entropy", max_depth=depth)

        # get error
        train_err, test_err = error(dclf, X, y)
        training_errors.append(train_err)
        testing_errors.append(test_err)
    # plt.plot(depths, training_errors)
    # plt.show()
    # plt.plot(depths, testing_errors)
    # plt.show()
    depth_to_test_err = dict(zip(depths, testing_errors))
    best_depth, lowest_error = -1, np.inf
    for k, v in depth_to_test_err.items():
        if v < lowest_error:
            best_depth, lowest_error = k, v
    print "best depth found: " + str(best_depth) + " which has error" + str(lowest_error)



    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: investigate decision tree classifier with various training set sizes
    print 'Investigating training set sizes...'
    splits = np.arange(0.05, 1,0.05)
    train_errors, test_errors = [], []
    for split in splits:
        # decision tree with previously best found depth
        dclf = DecisionTreeClassifier(criterion="entropy", max_depth=best_depth)
        train_err, test_err = error(dclf, X, y, test_size=split)
        train_errors.append(train_err)
        test_errors.append(test_err)
    # plt.plot(splits, train_errors)
    # plt.show()
    # plt.plot(splits, test_errors)
    # plt.show()
    split_error_dict = dict(zip(splits, test_errors))
    split, err = -1, np.inf
    for k, v in split_error_dict.items():
        if v < err:
            split, err = k, v
    print "best split found: " + str(split) + " which has test err: " + str(err)
    # super hyperparameter tuning
    possible_depths = np.arange(1,21)
    possible_training_splits = np.arange(0.05, 1, 0.05)
    possible_loss_funcs = ["gini", "entropy"]
    best_test_error, corresponding_training_error, best_depth, best_split, best_func = np.inf, np.inf, -1, -1, None
    for func in possible_loss_funcs:
        for depth in possible_depths:
            for tsplit in possible_training_splits:
                dclf = DecisionTreeClassifier(criterion=func, max_depth=depth)
                training_error, testing_error = error(dclf, X, y,
                                                      test_size=tsplit)
                if testing_error < best_test_error:
                    best_test_error, corresponding_training_error, best_depth, best_split, best_func = testing_error, training_error, depth, tsplit, func
    print "best testing error: " + str(best_test_error)
    print "corresponding_training_error: " + str(corresponding_training_error)
    print "best_depth: " + str(best_depth)
    print "best_split: " + str(best_split)
    print "best_func: " + str(best_func)




    ### ========== TODO : END ========== ###


    print 'Done'


if __name__ == "__main__":
    main()
