import numpy as np
import matplotlib.pyplot as plt


def plot(x, y):
    plt.plot(x, y, 'x')
    plt.axis('equal')
    plt.show()


def sample_from_bernoulli(p = 0.5):
    """Return 1 with probability p and 0 with probability 1 - p"""
    return 1 if np.random.rand() < p else 0

def compute_likelihood(data = [], p = 0.5):
    """Computes the likelihood of bernoulli distribution.
    Params: data: array of 0's and 1's, denoting the sample data.
            p: the probability of success (outcome of 1)
    Output: the likelihood of the data
    """
    likelihood = 1
    for x in data:
        likelihood*=(p if x ==1 else 1 - p)
    return likelihood

if __name__ == '__main__':
    data = [1,1,1,1,1,1,0,0,0,0]
    theta = np.arange(0.0,1.01,0.01)
    likelihood = [compute_likelihood(data, t) for t in theta]
    likelihood_dict = dict(zip(theta, likelihood))

    max_val = (-1, -1)
    for k, v in likelihood_dict.items():
        if v > max_val[1]:
            max_val = (k, v)
    print max_val
    # print likelihood
    # plot(theta, likelihood)
