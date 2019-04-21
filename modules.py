"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample

        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.

        Also, initialize gradients with zeros.
        """
        std = 0.0001
        mean = 0
        weights = std * np.random.randn(in_features, out_features) + mean

        weight_bias = np.zeros(out_features)

        grads = np.zeros((in_features, out_features))
        grad_bias = np.zeros(out_features)
        self.params = {'weight': weights, 'bias': weight_bias}
        self.grads = {'weight': grads, 'bias': grad_bias}
        self.previous_layer_feature_map = ""

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
        """

        self.feature_map = x
        out = x.dot(self.params['weight']) + self.params['bias']

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        dx = dout.dot(self.params['weight'].T)

        weight_gradients = dout.T.dot(self.feature_map)
        self.grads['weight'] = weight_gradients.T

        # since bias is just added in case of batch size 1, we add all of them at once
        bias_gradients = dout.sum(axis=0)

        self.grads['bias'] = bias_gradients
        return dx


class ReLUModule(object):
    """
    ReLU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
        """

        is_greater_than_zero = x > 0
        out = x * is_greater_than_zero
        self.feature_map = is_greater_than_zero

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        dx = dout * self.feature_map

        return dx


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
        """

        b = np.max(x, axis=1, keepdims=True)
        y = np.exp(x - b)

        out = y / y.sum(axis=1, keepdims=True)

        self.feature_map = out

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        size = self.feature_map.shape[1]
        batch_size = self.feature_map.shape[0]

        # compute diagonal matrix for all batches diagonal(x(N))
        diagonal_matrix = np.zeros((batch_size, size, size))
        diagonal_index = [i for i in range(0, size)]
        diagonal_matrix[:, diagonal_index, diagonal_index] = self.feature_map

        # x(N)^T*x(N) for all batches
        outer_vector_product = np.array(
            [np.reshape(self.feature_map[batch], [size, 1]).dot(np.reshape(self.feature_map[batch], [1, size]))
             for batch in range(batch_size)])

        right_hand_side_derivative = diagonal_matrix - outer_vector_product

        dx = np.array([dout[batch].dot(right_hand_side_derivative[batch]) for batch in range(batch_size)])

        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.

        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        losses = -np.log(np.sum(x * y, axis=1))

        return losses.mean()

    def backward(self, x, y):
        """
        Backward pass.

        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """
        batch_size = y.shape[0]  # the derivative of the bathc formula keeps 1/B constant
        dx = - y / x / batch_size
        return dx
