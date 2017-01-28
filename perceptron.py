# Implementation of the perceptron learning algorithm that linearly separates data.
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

class Perceptron(object):
    def __init__(self, num_params = 0):
        if num_params > 0:
            self.weights = np.random.uniform(low=0.0,high=1.0, size=num_params)

    def activation(self, x):
        return np.dot(self.weights, x)

    def sign_prediction(self, x):
        return 1 if self.activation(x) > 0 else -1

    def update_weights_if_needed(self, x, label):
        if self.activation(x) * label <= 0:
            self.weights = self.weights + label * x

    def shuffle_data(X, Y):
        features_plus_labels = np.zeros((X.shape[0], X.shape[1] + 1))
        features_plus_labels[:, :X.shape[1]] = X
        features_plus_labels[:, X.shape[1]] = Y
        np.random.shuffle(features_plus_labels)
        X = features_plus_labels[:, :X.shape[1]]
        Y = features_plus_labels[:, X.shape[1]]
        return X, Y

    def add_bias_unit(X):
        bias_added = np.ones((X.shape[0], X.shape[1] + 1))
        bias_added[:,:X.shape[1]] = X


    def train_model(self, X, Y, max_iter = 10000):
        # randomly initialize the weights
        self.weights = np.random.uniform(low=0.0, high=1.0, size=X.shape[1] + 1)
        # add a bias to X
        for epoch in max_iter:
            X, Y = shuffle_data(X, Y)
            for i in range(len(X)):
                feature_vector, label = X[i], Y[i]
                self.update_weights_if_needed(feature_vector, label)

        return self.weights

    def predict_labels(X):
        return [sign_prediction(feature_vector) for feature_vector in X]

    def get_error(predictions, labels):
        diffs = filter(lambda x: x!=0, [predictions - labels])
        return 100*len(diffs)/float(len(labels))

if __name__ == '__main__':
    plt.title("multi-class classification")
     # X1, Y1 = sklearn.datasets.make_classification(n_samples=500, n_features=10, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    X1, Y1 = sklearn.datasets.make_classification(n_samples = 500, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1, n_classes=2)
    plt.scatter(X1[:, 0], X1[:, 1], marker = 'o', c=Y1)
    plt.show()
    print Y1
