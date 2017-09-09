w = [2,-3,-3] # assume some random weights and data
x = [-1, -2]

w = [2, -3]
b = -1
x = [5, 6]
y = 1 # 1.5 * x1 - 1.08 * x2 + b
import numpy as np

for i in range(100):
	lin = w[0] * x[0] + w[1] * x[1] + b
	f = 1.0 / (1 + np.exp(-lin))
	prediction = 1 if f >= 0.5 else 0
	print("prediction: {}".format(prediction))
	e = (y - f)**2
	dedf = 2 * (y - f)
	dedlin = dedf * f * (1 - f) * -1 # (de/df) * (df/dlin) basically incoming gradient times local gradient
	dedw0 = x[0] * dedlin
	dedw1 = x[1] * dedlin
	dedb = 1.0 * dedlin
	w[0] -= 0.01 * dedw0
	w[1] -= 0.01 * dedw1
	print("dedw1: {}".format(dedw1))
	b -= 0.01 * dedb
# print("w0 grad: {}".format(dfdw0))
# print("w1 grad: {}".format(dfdw1))
# print("b1 grad: {}".format(dfdb1))