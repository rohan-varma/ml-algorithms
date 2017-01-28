# Implementation of the perceptron learning algorithm that linearly
# separates data.
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
class Perceptron(object):
    """Implementation of the perceptron algorithm for binary classification."""
    def __init__(self, num_params = 0):
        if num_params > 0:
            self.weights = np.random.uniform(low=0.0,high=1.0, size=num_params)

    def activation(self, x):
        """Computes the activation of the single unit by taking the dot product
        of the weights with the features
        """
        if len(self.weights) != len(x):
            raise AttributeError("incorrect dimensions. Received "
                                 + str(len(x)) + " input dim, but "
                                 + str(len(self.weights)) + " parameters")
        return np.dot(self.weights, x)

    def sign_prediction(self, x):
        """Applies the sign() function the the activation"""
        return 1 if self.activation(x) > 0 else -1


    def update_weights_if_needed(self, x, label):
        """Implementation of perceptron "learning". If the prediction is wrong
        (signs differ), we update the weights.
        """
        if self.activation(x) * label <= 0:
            self.weights = self.weights + label * x
            return True, self.weights

        return False, self.weights

    def shuffle_data(self, X, Y):
        """Shuffles the dataset to prevent highly correlated examples."""
        features_plus_labels = np.zeros((X.shape[0], X.shape[1] + 1))
        features_plus_labels[:, :X.shape[1]] = X
        features_plus_labels[:, X.shape[1]] = Y
        np.random.shuffle(features_plus_labels)
        X = features_plus_labels[:, :X.shape[1]]
        Y = features_plus_labels[:, X.shape[1]]
        return X, Y

    def add_bias_unit(self, X):
        """Adds the bias unit to the inputs"""
        bias_added = np.ones((X.shape[0], X.shape[1] + 1))
        bias_added[:,:X.shape[1]] = X
        return bias_added

    def train_model(self, X, Y, max_iter = 10000, vote = True, print_info = True):
        """Trains our perceptron classifier. Repeatedly iterates over the data,
        updating the weights when an incorrect classification is made.
        Params:
        X - input data
        Y - labels of input data
        max_iter: number of items to iterate over entire dataset
        print_info: if true, prints the error rate every 1k iterations.
        """
        # randomly initialize the weights
        self.weights = np.random.uniform(low=0.0, high=1.0, size=X.shape[1] + 1)
        # add a bias to X
        X = self.add_bias_unit(X)
        for epoch in range(max_iter):
            # print info every 1k iterations
            if print_info and epoch%1000 == 0:
                predictions = self.predict_labels(X)
                err = self.get_error(predictions, Y)
                print "current error: " + str(err)

            X, Y = self.shuffle_data(X, Y)
            for i in range(len(X)):
                feature_vector, label = X[i], Y[i]
                self.update_weights_if_needed(feature_vector, label)

        return self.weights

    def predict_labels(self, X):
        """Outputs a list of predictions for all input feature vectors."""
        return [self.sign_prediction(feature_vector) for feature_vector in X]

    def get_error(self, predictions, labels):
        """Computes the error by taking the difference between our predictions
        and labels.
        """
        diffs = filter(lambda x: x!=0, predictions - labels)
        return 100*len(diffs)/float(len(labels))

if __name__ == '__main__':
    # create and plot a dataset
    # plt.title("multi-class classification")
    X1, Y1 = sklearn.datasets.make_classification(n_samples = 500, n_features=2,
                                                  n_redundant=0,
                                                  n_informative=2,
                                                  random_state=1,
                                                  n_clusters_per_class=1,
                                                  n_classes=2)
    # plt.scatter(X1[:, 0], X1[:, 1], marker = 'o', c=Y1)
    # plt.show()
    # prepare data for classification
    Y1 = map(lambda y: -1 if y == 0 else 1, Y1)
    X1, Y1 = np.array(X1), np.array(Y1)
    perceptron = Perceptron(num_params = X1.shape[1])
    print "initial error:" + str(perceptron.get_error
                                 (perceptron.predict_labels(X1), Y1))
    print "training model"
    perceptron.train_model(X1, Y1)
    X1 = perceptron.add_bias_unit(X1)
    print "final error:" + str(perceptron.get_error
                               (perceptron.predict_labels(X1), Y1))
