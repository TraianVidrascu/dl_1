"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import pickle as pk

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

import torch
import torch.nn as nn
import torch.nn.functional as F


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    class_values, classes = torch.max(predictions, 1)
    correct_predicted = targets == classes
    accuracy = correct_predicted.type(torch.float).mean()
    return accuracy


def train():
    """
    Performs training and evaluation of ConvNet model.

    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    torch.manual_seed(42)

    device = torch.device("cuda")

    conv_net = ConvNet(3, 10)
    conv_net.to(device)

    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)

    # get train data
    x_train, y_train = cifar10["train"].images, cifar10["train"].labels
    x_train = torch.torch.from_numpy(x_train).to(device)

    y_train = y_train.argmax(axis=1)
    y_train = torch.from_numpy(y_train).to(device).type(torch.long)

    # get test data
    x_test, y_test = cifar10["test"].images, cifar10["test"].labels
    inputs_test = torch.torch.from_numpy(x_test).to(device)

    y_test = y_test.argmax(axis=1)
    targets_test = torch.from_numpy(y_test).to(device).type(torch.long)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(conv_net.parameters(), lr=FLAGS.learning_rate)

    losses_test = []
    losses_train = []
    accuracies_test = []
    accuracies_train = []
    for i in range(FLAGS.max_steps):
        # getting batch for forwarding
        x, y = cifar10['train'].next_batch(FLAGS.batch_size)

        class_indexes_targets = y.argmax(axis=1)  # pytorch does not support one hot encoded vectors as targests
        # making a tensor which pytorch can work with
        inputs = torch.from_numpy(x).to(device)
        targets = torch.from_numpy(class_indexes_targets).to(device).type(torch.long)

        # making gradients zero, framework tutorial states if we don't do so the gradients at each step will just add
        optimizer.zero_grad()

        predictions = conv_net.forward(inputs)  # forwarding batch into the network
        loss = loss_function.forward(predictions, targets)  # calculating the Cross Entropy loss
        loss.backward()  # backwards the loss into the net, updating gradients

        optimizer.step()  # updating the weights

        if i % FLAGS.eval_freq == 0 or i == FLAGS.max_steps - 1:
            # evaluating test
            forward_test = conv_net.forward(inputs_test)

            acc = accuracy(forward_test, targets_test)
            accuracies_test.append(acc.item())

            loss_test = loss_function.forward(forward_test, targets_test)
            losses_test.append(loss_test.item())
            print(
                "TEST loss:" + str(round(losses_test[-1], 2)) + " acc:" + str(
                    round(accuracies_test[-1], 2)) + " model:" + str(i))

            # evaluating train
            forward_train = conv_net.forward(x_train)

            acc_train = accuracy(forward_train, y_train)
            accuracies_train.append(acc_train.item())

            loss_train = loss_function.forward(forward_train, y_train)
            losses_train.append(loss_train.item())
            print(
                "TRAIN loss:" + str(round(losses_train[-1], 2)) + " acc:" + str(
                    round(accuracies_train[-1], 2)) + " model:" + str(i))

    with open('../results/torch_conv.pkl', 'wb') as f:
        mlp_data = dict()
        mlp_data["train_loss"] = losses_train
        mlp_data["test_loss"] = losses_test
        mlp_data["train_acc"] = accuracies_train
        mlp_data["test_acc"] = accuracies_test
        pk.dump(mlp_data, f)


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()
