import numpy as np

# random date
x = np.random.randn(50, 100) # 50 examples of 100 dim
y = np.random.randn(50, 10) # 50 targets, where there are 10 class labels
w1 = np.random.randn(100, 500)
w2 = np.random.randn(500, 10)
learning_rate = 1e-6

def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    x[x <= 0] = 0
    x[x>0] = 1
    return x

for t in range(500):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2