import numpy as np


def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.

    The input x has shape (N, Din) and contains a minibatch of N
    examples, where each example x[i] has shape (Din,).

    Inputs:
    - x: A numpy array of shape (N, Din) giving input data
    - w: A numpy array of shape (Din, Dout) giving weights
    - b: A numpy array of shape (Dout,) giving biases

    Returns a tuple of:
    - out: output, of shape (N, Dout)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the forward pass. Store the result in out.              #
    ###########################################################################
    ###########################################################################
     # From equation (1.2) in the image: y_i = x_i W + b

    out = np.dot(x, w) + b
    
        # cache inputs forn the backward pass
    cache = (x, w, b)
    return out, cache
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def fc_backward(grad_out, cache):
    """
    Computes the backward pass for a fully-connected layer.

    Inputs:
    - grad_out: Numpy array of shape (N, Dout) giving upstream gradients
    - cache: Tuple of:
      - x: A numpy array of shape (N, Din) giving input data
      - w: A numpy array of shape (Din, Dout) giving weights
      - b: A numpy array of shape (Dout,) giving biases

    Returns a tuple of downstream gradients:
    - grad_x: A numpy array of shape (N, Din) of gradient with respect to x
    - grad_w: A numpy array of shape (Din, Dout) of gradient with respect to w
    - grad_b: A numpy array of shape (Dout,) of gradient with respect to b
    """
    x, w, b = cache
    grad_x, grad_w, grad_b = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for the fully-connected layer         #
    ###########################################################################
    ###########################################################################
    x, w, b = cache
    
    # From equation (1.9): dL/dx = dL/dy * W^T
  
  
    grad_x = np.dot(grad_out, w.T)
    
    # From equation (1.15): dL/dW = x^T * dL/dy
    grad_w = np.dot(x.T, grad_out)
    
    # From equation (1.21): dL/db = 1^T * dL/dy
         # Sum across all examples (axis 0) to get the gradient for each bias term
    grad_b = np.sum(grad_out, axis=0)
    
    return grad_x, grad_w, grad_b
    #                             END OF YOUR CODE                            #
  
    ###########################################################################
    return grad_x, grad_w, grad_b


def relu_forward(x):
    """
    Computes the forward pass for the Rectified Linear Unit (ReLU) nonlinearity

    Input:
    - x: A numpy array of inputs, of any shape

    Returns a tuple of:
    - out: A numpy array of outputs, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    ###########################################################################
    
    out = np.maximum(0, x)
    cache = x
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(grad_out, cache):
    """
    Computes the backward pass for a Rectified Linear Unit (ReLU) nonlinearity

    Input:
    - grad_out: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - grad_x: Gradient with respect to x
    """
    grad_x, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    ###########################################################################
    x = cache
   
    grad_x = grad_out * (x > 0)
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return grad_x


def l2_loss(x, y):
    """
    Computes the loss and gradient of L2 loss.

    loss = 0.5 * sum_i (x_i - y_i)**2 / N

    Inputs:
    - x: Input data, of shape (N, D)
    - y: Output data, of shape (N, D)

    Returns a tuple of:
    - loss: Scalar giving the loss
    - grad_x: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    diff = x - y
    loss = 0.5 * np.sum(diff * diff) / N
    grad_x = diff / N
    return loss, grad_x


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax (cross-entropy) loss function.

    Inputs:
    - x: Numpy array of shape (N, C) giving predicted class scores, where
      x[i, c] gives the predicted score for class c on input sample i
    - y: Numpy array of shape (N,) giving ground-truth labels, where
      y[i] = c means that input sample i has ground truth label c, where
      0 <= c < C.

    Returns a tuple of:
    - loss: Scalar giving the loss
    - grad_x: Numpy array of shape (N, C) giving the gradient of the loss with
      with respect to x
    """
    loss, grad_x = None, None
    ###########################################################################
    # TODO: Implement softmax loss                                            #
    ###########################################################################
    ###########################################################################
    N = x.shape[0]
    
    # Forward pass
    # Subtract maximum for numerical stability (eq 1.26)
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_logits)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # 
    #Compute loss (eq 1.25)
    correct_logprobs = -np.log(probs[np.arange(N), y])
    loss = np.sum(correct_logprobs) / N
    
    #  Backward pass (eq 1.35)
    grad_x = probs.copy()
    grad_x[np.arange(N), y] -= 1
    grad_x /= N
    
    return loss, grad_x
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, grad_x


def l2_regularization(w, reg):
    """
    Computes loss and gradient for L2 regularization of a weight matrix:

    loss = (reg / 2) * sum_i w_i^2

    Where the sum ranges over all elements of w.

    Inputs:
    - w: Numpy array of any shape
    - reg: float giving the regularization strength

    Returns:
    """
    loss, grad_w = None, None
    ###########################################################################
    # TODO: Implement L2 regularization.                                      #
    ###########################################################################
    ###########################################################################
 # Compute loss: (reg/2) * sum(w_i^2)
    loss = 0.5 * reg * np.sum(w * w)
    
     # Compute gradient: reg * w
    grad_w = reg * w
    
    return loss, grad_w
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, grad_w
