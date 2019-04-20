"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *

class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """
        inputs = n_inputs
        self.modules = []
        for n_layer in n_hidden:
            linear_module = LinearModule(inputs, n_layer)
            activation_module = ReLUModule()
            inputs = n_layer
            self.modules.append(linear_module)
            self.modules.append(activation_module)

        output_layer = LinearModule(inputs, n_classes)
        soft_max_module = SoftMaxModule()

        self.modules.append(output_layer)
        self.modules.append(soft_max_module)

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        out = x
        for layer in self.modules:
            out = layer.forward(out)

        return out

    def backward(self, dout):
        for module in reversed(self.modules):
            dout = module.backward(dout)
