import numpy as np
from utils import *
class SVM(object):
    def __init__(self):
        pass


    def loss(y, x, weights, bias, l2_reg = 0, l1_reg = 0):
        """Implementation of regularized hinge loss for SVM
        Params:
        y - target label (if multiclass, 1 x n where index n is the label)
        x - single input feature vector, 1 x d
        w - weights, n x d where n = number of output classes
        bias - bias vector: 1 x n (scalar if n = 1)
        l2_reg: l2 regularization constant
        l1_reg: l1 regularization constant
        Returns:
        reg_hinge_loss: the hinge loss with a regularization term
        """
        mult = (weights.T).dot(x) + bias
        hinge_loss = max(0, 1 - labels * mult)
        reg_term = l1_reg * weights + l2_reg * (np.linalg.norm(weights)**2)
        return hinge_loss + reg_term

    def vectorized_loss(y, x, weights, bias, l2_reg = 0, l1_reg = 0):
        """Fully vectorized regularized hinge loss implementation
        Params: 
        y
        """
        raise(NotImplementedError())

