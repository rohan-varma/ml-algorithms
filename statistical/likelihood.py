import numpy as np
import matplotlib.pyplot as plt

def sample_from_bernoulli(p = 0.5):
    """Return 1 with probability p and 0 with probability 1 - p"""
    return 1 if np.random.rand() < p else 0

def compute_likelihood(data = [], p = 0.5):
    """Computes the likelihood of bernoulli distribution.
    Params: data: array of 0's and 1's, denoting the sample data.
            p: the probability of success (outcome of 1)
    Returns: the likelihood of the data
    """
    likelihood = 1
    for x in data:
        likelihood*=(p if x ==1 else 1 - p)
    return likelihood

def get_max_val(d):
    """ return tuple of max val in dict, assumes vals are >= 0"""
    max_val = (-1, -1)
    for k, v in d.items():
        if v > max_val[1]:
            max_val = k, v
    return max_val

def generate_likelihood(data, theta):
    """Computes the likelihood of the data assuming it is sampled from the
    bernoulli distribution for a list of parameters theta.
    Params:
        data: array-like of 0/1
        theta: list of parameters (0<= p <= 1) for which to compute likelihood
    Returns:
        likelihood: a list of the likelihoods corresponding to the parameters
        likelihood_dict: a dict of likelihood: parameter pairs
        max_val: the k, v with maximum likelihood.
    """
    likelihood = [compute_likelihood(data, t) for t in theta]
    likelihood_dict = dict(zip(theta, likelihood))
    max_val = get_max_val(likelihood_dict)
    return likelihood, likelihood_dict, max_val


if __name__ == '__main__':
    data = [1,1,1,1,1,1,0,0,0,0]
    theta = np.arange(0.0,1.01,0.01)
    likelihood, likelihood_dict, max_val = generate_likelihood(data, theta)
    plt.plot(theta, likelihood)
    plt.show()
    print "max for 6 ones: " + str(max_val)

    # 111 00 plot

    data = [1,1,1,0,0]
    likelihood, likelihood_dict, max_val = generate_likelihood(data, theta)
    plt.plot(theta, likelihood)
    plt.show()
    print "max for 3 ones" + str(max_val)

    # 60 ones plot
    data = np.ones(100)
    data[60:] = 0
    likelihood, likelihood_dict, max_val = generate_likelihood(data, theta)
    plt.plot(theta, likelihood)
    plt.show()
    print "Max for 60 ones: " + str(max_val)

    # 5 ones plot
    data = [1,1,1,1,1,0,0,0,0,0]
    likelihood, likelihood_dict, max_val = generate_likelihood(data, theta)
    plt.plot(theta, likelihood)
    plt.show()
    print "max for 5 ones: " + str(max_val)
