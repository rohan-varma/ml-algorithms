"""Implementation of several loss functions"""
import numpy as np

def zero_one_loss(y, pred, average = True):
    """Computes the zero-one loss"""
    if y.shape[0] != pred.shape[0]:
        raise AttributeError("Dimensions of input vectors should be the same")
    return np.sum(y - pred)/y.shape[0] if average else np.sum(y - pred)

def cross_entropy_loss(y, pred, average = True):
    """Computes the negative log likelihood for binary logistic regression"""
    if y.shape[0] != pred.shape[0]:
        raise AttributeError("Dimensions of input vectors should be the same")
    nll =  -np.sum(y*np.log(pred) + (1-y)*np.log(1-pred))
    return nll/y.shape[0] if average else nll

def softmax_multinomial_loss(y, pred, average = True):
    """Computes the generalized logistic loss"""
    if y.shape[0] != pred.shape[0]:
        raise AttributeError("Dimensions of input vectors should be the same")
    nll = -np.sum(y*np.log(pred))
    return nll/y.shape[0] if average else nll

def hinge_loss(y, pred, average = True):
    """Computes the hinge loss for SVMs"""
    if y.shape[0] != pred.shape[0]:
        raise AttributeError("Dimensions of input vectors should be the same")
    hl = 1-(y*pred)
    hl = filter(lambda k: k > 0, hl)
    return np.sum(hl)/y.shape[0] if average else np.sum(hl)

def squared_l2_loss(y, pred, average = True):
    """Computes the squared error for regression"""
    if y.shape[0] != pred.shape[0]:
        raise AttributeError("Dimensions of input vectors should be the same")
    squared_norm = np.square(np.linalg.norm(y - pred))
    return squared_norm/y.shape[0] if average else squared_norm
