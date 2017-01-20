import numpy as np
import matplotlib.pyplot as plt

def plot(x, y):
    plt.plot(x, y, 'x')
    plt.axis('equal')
    plt.show()

mean = [0, 0]
# cov = [[1, 0], [0, 1]]
# x, y = np.random.multivariate_normal(mean, cov, 1000).T

# now try a mean of [1, 1]
# x, y = np.random.multivariate_normal(mean, cov, 1000).T
# plot(x, y)
cov = [[1, -0.5], [-0.5, 1]]
x, y = np.random.multivariate_normal(mean, cov, 1000).T
plot(x, y)
