import numpy as np
from layers import fc_forward, fc_backward, relu_forward, relu_backward, l2_loss, softmax_loss, l2_regularization

# Sample inputs for testing
N, Din, Dout = 4, 5, 3
grad_out = np.random.randn(N, Dout)  # Upstream gradients
x = np.random.randn(N, Din)          # Input data
w = np.random.randn(Din, Dout)       # Weights
b = np.random.randn(Dout)            # Biases
y_true = np.random.randint(0, Dout, N)  # True labels for softmax_loss

# Test fc_forward
out_fc_forward, cache_fc_forward = fc_forward(x, w, b)
print("fc_forward output:", out_fc_forward)

# Test fc_backward
grad_x, grad_w, grad_b = fc_backward(grad_out, cache_fc_forward)
print("fc_backward grad_x:", grad_x)
print("fc_backward grad_w:", grad_w)
print("fc_backward grad_b:", grad_b)

# Test relu_forward
out_relu, cache_relu = relu_forward(x)
print("relu_forward output:", out_relu)

# Test relu_backward
grad_relu = relu_backward(grad_out, cache_relu)
print("relu_backward grad:", grad_relu)

# Test l2_loss
l2_loss_value = l2_loss(w)
print("l2_loss:", l2_loss_value)

# Test softmax_loss
loss_softmax, grad_softmax = softmax_loss(out_fc_forward, y_true)
print("softmax_loss:", loss_softmax)
print("softmax_loss gradient:", grad_softmax)

# Test l2_regularization
l2_reg_loss, grad_l2_reg = l2_regularization(w)
print("l2_regularization loss:", l2_reg_loss)
print("l2_regularization gradient:", grad_l2_reg)
