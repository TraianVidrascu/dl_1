"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem


        TODO:
        Implement initialization of the network.
        """

        super(ConvNet, self).__init__()

        #1111111111111
        self.conv1 = nn.Conv2d(n_channels, 64, 3, 1, 1)
        self.layers = nn.ModuleList([self.conv1])

        self.batch_1 = nn.BatchNorm2d(64)
        self.layers.append(self.batch_1)

        self.relu_1 = nn.ReLU()
        self.layers.append(self.relu_1)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers.append(self.max_pool_1)


        #222222222222

        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, stride=1, padding=1)
        self.layers.append(self.conv2)

        self.batch_2 = nn.BatchNorm2d(128)
        self.layers.append(self.batch_2)

        self.relu_2 = nn.ReLU()
        self.layers.append(self.relu_2)

        self.max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers.append(self.max_pool_2)


        # 333333333

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.layers.append(self.conv3)

        self.batch_3 = nn.BatchNorm2d(256)
        self.layers.append(self.batch_3)

        self.relu_3 = nn.ReLU()
        self.layers.append(self.relu_3)

        self.max_pool_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers.append(self.max_pool_3)

        # 4444444444
        # aaaaaaaaaa
        self.conv4_a = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.layers.append(self.conv4_a)

        self.batch_4_a = nn.BatchNorm2d(512)
        self.layers.append(self.batch_4_a)

        self.relu_4_a = nn.ReLU()
        self.layers.append(self.relu_4_a)

        # bbbbbbb
        self.conv4_b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.layers.append(self.conv4_b)

        self.batch_4_b = nn.BatchNorm2d(512)
        self.layers.append(self.batch_4_b)

        self.relu_4_b = nn.ReLU()
        self.layers.append(self.relu_4_b)

        self.max_pool_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers.append(self.max_pool_4)

        #555555555555555
        # aaaaaaaaaa
        self.conv5_a = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.layers.append(self.conv5_a)

        self.batch_5_a = nn.BatchNorm2d(512)
        self.layers.append(self.batch_5_a)

        self.relu_5_a = nn.ReLU()
        self.layers.append(self.relu_5_a)

        # bbbbbbbbb
        self.conv5_b = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.layers.append(self.conv5_b)

        self.batch_5_b = nn.BatchNorm2d(512)
        self.layers.append(self.batch_5_b)

        self.relu_5_b = nn.ReLU()
        self.layers.append(self.relu_5_b)

        self.max_pool_5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers.append(self.max_pool_5)

        self.avg_pool = nn.AvgPool2d(kernel_size=1,stride=1,padding=0)
        self.layers.append(self.avg_pool)

        self.linear = nn.Linear(in_features=512, out_features=n_classes)
        self.layers.append(self.linear)

        self.soft_max = nn.Softmax()
        self.layers.append(self.soft_max)

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
        for layer in self.layers:
            if isinstance(layer,nn.Linear):
                out = torch.squeeze(out)

            out = layer.forward(out)


        return out
