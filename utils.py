"""Utils to perform error checking, CV, and hyperparameter tuning."""
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import sklearn.datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
# 1- metrics.accuracy_score(y_test, y_pred_test, normalize=True)

# TODO K-Fold CV for training/testing error
# TODO K-Fold CV for hyperparameter tuning

def split_data(X, y, k = 10):
    """Splits data"""
    X_split = np.array_split(X, k)
    y_split = np.array_split(y, k)
    return X_split, y_split

def cross_validate(classifier, X, y, k = 10):
    X_split, y_split = split_data(X, y, k)
    # for every k, train & evaluate a classifier
    training_errors, testing_errors = [], []
    for i in range(k):
        X_test, y_test = X_split[i], y_split[i]
        X_train = np.concatenate([X_split[j] for j in range(len(X_split)) if j!=i])
        y_train = np.concatenate([y_split[j] for j in range(len(y_split)) if j!=i])
        # train and test the model
        train_error, test_error = get_errors_already_split(classifier, X_train,
                                                           y_train, X_test,
                                                           y_test,
                                                           num_iterations=1)
        training_errors.append(train_error)
        testing_errors.append(test_error)

    mean_train_error = np.mean(np.array(training_errors), axis=0)
    mean_test_error = np.mean(np.array(testing_errors), axis=0)
    return mean_train_error, mean_test_error





def get_errors_already_split(classifier, X_train, y_train, X_test, y_test, num_iterations=100):
    train_error, test_error = 0.0, 0.0
    for i in range(num_iterations):
        classifier.fit(X_train, y_train)
        y_train_pred = classifier.predict(X_train)
        y_test_pred = classifier.predict(X_test)
        train_error+=1- metrics.accuracy_score(y_true=y_train,
                                               y_pred=y_train_pred,
                                               normalize=True)
        test_error+=1 - metrics.accuracy_score(y_true=y_test,
                                               y_pred=y_test_pred,
                                               normalize=True)
    train_error /=num_iterations
    test_error /=num_iterations
    return train_error, test_error


def get_train_test_error(classifier, X, y, num_iterations = 100, split = 0.2):
    """Returns the average error over num_iterations"""
    train_error, test_error = 0.0, 0.0
    for i in range(num_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=split,
                                                            random_state=i)
        classifier.fit(X_train, y_train)
        y_train_pred = classifier.predict(X_train)
        y_test_pred = classifier.predict(X_test)
        train_error+=1- metrics.accuracy_score(y_true=y_train,
                                               y_pred=y_train_pred,
                                               normalize=True)
        test_error+=1 - metrics.accuracy_score(y_true=y_test,
                                               y_pred=y_test_pred,
                                               normalize=True)
    train_error /=num_iterations
    test_error /=num_iterations
    return train_error, test_error


def get_best_depth(classifier, X, y, depths = [8]):
    best_test_err, corr_train_err, best_depth = np.inf, -1, -1
    for depth in depths:
        d_tree = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
        train_err, test_err = get_train_test_error(classifier, X, y)
        if test_err < best_test_err:
            best_test_err, corr_train_err, best_depth = test_err, train_err, depth
    return best_test_err, corr_train_err, best_depth





# test it with decision tree
print "creating dataset"
X, y = sklearn.datasets.make_classification(n_samples = 800, n_features=10,
                                              n_redundant=0,
                                              n_informative=10,
                                              random_state=0,
                                              n_clusters_per_class=1,
                                              n_classes=2)

X, y = np.array(X), np.array(y)
d_tree = DecisionTreeClassifier(criterion="entropy")
print "training & evaluating decision tree"
train_err, test_err = get_train_test_error(d_tree, X, y, split=0.7)
print "training error: " + str(train_err)
print "testing error: " + str(test_err)
print "getting cross validation errors"
train_error_cv, test_error_cv = cross_validate(classifier=d_tree, X=X, y=y, k = 799)
print "training CV error: " + str(train_error_cv)
print "testing cv error: " + str(test_error_cv)

print "trying to find best depth...."
depths = np.arange(1,21)
best_test_err, corr_train_err, best_depth = get_best_depth(classifier=d_tree, X=X, y=y, depths = depths)
print "best depth found: " + str(best_depth)
print "testing error for that: " + str(best_test_err)
