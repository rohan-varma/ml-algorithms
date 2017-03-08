import numpy as np
class FeatureTransformer(object):
    """A utility for generating nonlinear features during preprocessing"""
    def __init__(self, degree, decay = None):
        """initialize the transformer.
        Params:
            degree: The degree d of polynomial features that are needed.
            For example, if d = 2 and x = [1, x_1, x_2] then the transformation
            will be x = [1, x_1, x_2, x_1^2, x_1 * x_2, x_2 * x_1, x_2^2]
            decay: if given, each new feature will be decayed by a power of decay that increases with d.
            Higher-degree features will thus play less of a role in the model.
        """
        self.degree = degree
        self.decay = decay

    def apply_decay(self, X):
        """Params: X: numpy matrix where each row is a feature vector that is (possibly) transformed"""
        decay_vector = np.array([self.decay**r for r in np.arange(0, X.shape[0] + 0.5, 0.5)])
        decay_vector = decay_vector[:X.shape[1] if len(X.shape) == 2 else X.shape[0]]
        if len(X.shape) == 2:
            assert(decay_vector.shape[0] == X.shape[1])
        else:
            assert(decay_vector.shape[0] == X.shape[0])
        return X*decay_vector


    def generate_features(self, X):
        """Generates the features"""
        if len(X.shape) == 1:
            X_n = np.ones((1, X.shape[0]))
            X_n[0] = X[0]
            X = X_n
        X_new = []
        for row in X:
            cur = list(row)
            new_row = [1]
            new_row.append(cur)
            if self.degree > 1:
                for d in range(2, self.degree+1):
                    for i in range(len(cur)):
                        li = [(cur[i]**d) * k for k in cur]
                        new_row.append(li)
                # flattened = [val for sublist in new_row for val in sublist]
            print new_row
            X_new.append(new_row)
        return np.array(X_new)
