import numpy as np
class SVM(object):
    def __init__(self):
        pass

    def loss(label, x, weights, bias):
        """Implementation of regularized hinge loss for
            a single training example, label pair"""
        # compute predictions
        mult = (weights.T).dot(x) + bias
        # mult is a vector of class label scores
        # labels are binary
        hinge_loss = max(0, 1 - labels*mult)
