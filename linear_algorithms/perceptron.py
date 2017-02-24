# Implementation of the perceptron learning algorithm that linearly
# separates data.
import numpy as np
import sklearn.datasets

class Perceptron(object):
    """Implementation of the perceptron algorithm for binary classification."""
    def __init__(self, num_params=0):
        if num_params > 0:
            self.weights = np.random.uniform(low=0.0,high=1.0,
                                             size=num_params+1)

    def activation(self, x):
        """Computes the activation of the single unit by taking the dot product
        of the weights with the features.
        """
        if len(self.weights) != len(x):
            raise AttributeError("incorrect dimensions. Received "
                                 + str(len(x)) + " input dim, but "
                                 + str(len(self.weights)) + " parameters")
        return np.dot(self.weights, x)

    def sign_prediction(self, x):
        """Applies the sign() function to the activation"""
        return 1 if self.activation(x) > 0 else -1

    def weighted_prediction(self, X):
        """Makes a binary prediction using many perceptrons. Each perceptron is
        weighted by the number of times it has "survived", or how many times
        it has correctly classified an example in a row. The prediction is:
        sign(sum(perceptron_survival_weighted * sign(perceptron * input)))
        """
        print "doing weighted prediction"
        predictions = [1 if sum([perceptron[1] * (1 if np.dot(perceptron[0], x)
                                                  > 0 else -1) for perceptron in
                                                  self.perceptrons]) > 0 else -1
                                                  for x in X]

        # the above list comprehension is equivalent to the following code
        # predictions = []
        # for x in X:
        #     cur_sum = sum([perceptron[1] * (1 if np.dot(perceptron[0], x) > 0
        #                                     else -1) for perceptron in
        #                                      self.perceptrons])
        #     predictions.append(1 if cur_sum > 0 else -1)
        return predictions

    def update_weights_if_needed(self, x, label):
        """Implementation of perceptron "learning". If the prediction is wrong
        (signs differ), we update the weights and return false. Otherwise the
        weights are not updated and we return True.
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

    def print_info(self, X, Y):
        predictions = self.predict_labels(X)
        err = self.get_error(predictions, Y)
        print "error for current perceptron: " + str(err)


    def train_model(self, X, Y, max_iter = 10000, vote = True,
                    print_info = True, survived_threshold = 1):
        """Trains our perceptron classifier. Repeatedly iterates over the data,
        updating the weights when an incorrect classification is made.
        Params:
        X - input data
        Y - labels of input data
        max_iter: number of items to iterate over entire dataset
        print_info: if true, prints the error rate every 1k iterations.
        """
        # randomly initialize the weights
        if len(self.weights) != X.shape[1] + 1:
            self.weights = np.random.uniform(low=0.0, high=1.0,
                                             size=X.shape[1] + 1)
        # create a list to hold how long the perceptron survived
        perceptron_survivied = []
        cur_survived = [self.weights, 0]
        # add a bias to X
        X = self.add_bias_unit(X)
        for epoch in range(max_iter):
            # print info every 1k iterations
            if print_info and epoch%1000 == 0: self.print_info(X, Y)
            X, Y = self.shuffle_data(X, Y)
            for i in range(len(X)):
                feature_vector, label = X[i], Y[i]
                if vote:
                    # use voting algorithm
                    weights_updated, weights = \
                    self.update_weights_if_needed(feature_vector, label)
                    if weights_updated:
                        # possibly save the perceptron if its performed well
                        if cur_survived[1] >= survived_threshold:
                            perceptron_survivied.append(cur_survived)
                        cur_survived = [self.weights, 0]
                    else:
                        # the perceptron classified correctly, so update
                        cur_survived[1]+=1
                else:
                    self.update_weights_if_needed(feature_vector, label)
        # add the last weights
        perceptron_survivied.append(cur_survived)

        if vote:
            self.perceptrons = perceptron_survivied
            return self.weights, self.perceptrons
        else:
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
    X, Y = sklearn.datasets.make_classification(n_samples = 800, n_features=10,
                                                  n_redundant=0,
                                                  n_informative=10,
                                                  random_state=0,
                                                  n_clusters_per_class=1,
                                                  n_classes=2)
    # plt.scatter(X1[:, 0], X1[:, 1], marker = 'o', c=Y1)
    # plt.show()
    # prepare data for classification
    X1, Y1, X2, Y2 = [], [], [], []
    for i in range(len(X)):
        rand_n= np.random.random_integers(low=1, high=10)
        if rand_n < 8:
            X1.append(X[i])
            Y1.append(Y[i])
        else:
            X2.append(X[i])
            Y2.append(Y[i])

    #X1, Y1, X2, Y2 = X1[0:500], Y1[0:500], X1[500:], Y1[500:]
    Y1 = map(lambda y: -1 if y == 0 else 1, Y1)
    X1, Y1 = np.array(X1), np.array(Y1)
    X2, Y2 = np.array(X2), np.array(Y2)
    threshold, iters = 15, 10000
    print "training model"
    perceptron.train_model(X1, Y1, max_iter=10000, survived_threshold = 40)
    X1, X2 = perceptron.add_bias_unit(X1), perceptron.add_bias_unit(X2)
    print "final training error: " + str(perceptron.get_error
                               (perceptron.weighted_prediction(X1), Y1))
