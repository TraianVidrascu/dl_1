"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import pickle as pk

import matplotlib.pyplot as plt
from mlp_numpy import MLP, LinearModule
from modules import CrossEntropyModule
import cifar10_utils

# Default constants
IMAGE_SIZE = 32 * 32 * 3
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


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

    results = predictions.argmax(axis=1)  # get the positons of the maximum prediction from the softmax module
    actual = targets.argmax(axis=1)  # get the positions of the actual targets

    correct_results = results == actual

    accuracy = correct_results.sum() / correct_results.size

    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    # get test data set
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)

    x_test, y_test = cifar10["test"].images, cifar10["test"].labels
    x_test = x_test.reshape((x_test.shape[0], IMAGE_SIZE))

    mlp = MLP(IMAGE_SIZE, dnn_hidden_units, 10)

    loss = CrossEntropyModule()
    losses_test = []
    losses_train = []
    accuracies_test = []
    accuracies_train = []
    for i in range(FLAGS.max_steps):
        # getting batch for forwarding
        x, y = cifar10['train'].next_batch(FLAGS.batch_size)
        x = x.reshape((FLAGS.batch_size, IMAGE_SIZE))

        # forwarding batch
        forward_test = mlp.forward(x)
        loss_train = loss.forward(forward_test, y)

        # backpropagation of error
        loss_grad = loss.backward(forward_test, y)
        mlp.backward(loss_grad)

        # learning with SGD
        for module in mlp.modules:
            # updating for linear modules
            if isinstance(module, LinearModule):
                module.params['weight'] -= FLAGS.learning_rate * module.grads['weight']
                module.params['bias'] -= FLAGS.learning_rate * module.grads['bias']

        acc_train = accuracy(forward_test, y)

        losses_train.append(loss_train)
        accuracies_train.append(acc_train)
        if i % FLAGS.eval_freq == 0 or i == FLAGS.max_steps - 1:
            forward_test = mlp.forward(x_test)
            acc_test = accuracy(forward_test, y_test)
            loss_test = loss.forward(forward_test, y_test)
            print("acc: " + str(round(acc_test, 2)) + " loss: " + str(round(loss_test, 3)) + " ittr: " + str(i))
            losses_test.append(loss_test)
            accuracies_test.append(acc_test)

    with open('../results/numpy_mlp.pkl', 'wb') as f:
        mlp_data = dict()
        mlp_data["train_loss"] = losses_train
        mlp_data["test_loss"] = losses_test
        mlp_data["train_acc"] = accuracies_train
        mlp_data["test_acc"] = accuracies_test
        pk.dump(mlp_data, f)

    x = [i * FLAGS.eval_freq for i in range(len(accuracies_test))]
    plt.plot(x, accuracies_test, label="accuracy")
    plt.plot(x, losses_test, label="loss")
    plt.legend()
    plt.savefig("../results/numpy_mlp.png")
    plt.show()


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
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
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
