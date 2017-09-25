import numpy as np
import time
import matplotlib.pyplot as plt

w = [0, 0]
b = 0
x = [5, 6]
y = 1.00 # 1.5 * x1 - 1.08 * x2 - 0.02
predictions, errors = [], []
def sigmoid(x, deriv = False):
	s = 1 / (1 + np.exp(-x))
	return s if not deriv else s * (1 - s)

def relu(x, deriv = False):
	if deriv:
		if x == 0:
			return 0.5
		return 1 if x > 0 else 0
	return max(0, x)

def leaky_relu(x, deriv = False):
	alpha = 0.05
	if not deriv:
		return alpha * x if x <0 else x
	else:
		if x == 0:
			return 0.5
		return alpha if x < 0 else 1

nonlin_function = relu # change this to test different nonlinearities
start = time.clock()
eps = 0.001
for i in range(10000):
	assert len(predictions) == i and len(errors) == i
	lin = w[0] * x[0] + w[1] * x[1] + b
	prediction = nonlin_function(lin) # 
	print("prediction: {}".format(prediction))
	predictions.append(prediction)
	err = (y - prediction) ** 2
	errors.append(err)
	print("error: {}".format(err))
	if abs(y-prediction) < eps:
		break
	dedf = -2 * (y - prediction)
	dedl = dedf * nonlin_function(lin, deriv = True)
	dedw0 = x[0] * dedl
	dedw1 = x[1] * dedl
	dedb = 1 * dedl
	w[0] += -0.01 * dedw0
	w[1] += -0.01 * dedw1
	b += -0.01 * dedb
print("done at iteration {}".format(i))
print("final parameter values w0 w1 b: {} {} {}".format(w[0], w[1], b))
print("final prediction: {}".format(predictions[-1]))
print("final error: {}".format((y - predictions[-1]) **2))
end = time.clock()
print(end - start)
plt.plot(list(range(len(predictions))), predictions)
plt.plot(list(range(len(errors))), errors)
plt.show()