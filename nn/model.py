import tensorflow as tf
import numpy as np
import os
import progressbar
from datetime import datetime

IMG_SIZE_PX = 32

n_classes = 2
batch_size = 10

saved_graph_path = os.getcwd()+'/data/tmp/log/'

keep_rate = 0.8


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding="SAME")


def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(
        x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding="SAME"
    )


"""
This functions constructs the neural network.
@params x: placeholder
"""


def convolutional_neural_network(x, name="conv_neural_net"):
    with tf.name_scope(name):
        #                # 5 x 5 x 5 patches, 1 channel, 64 features to compute.
        weights = {
            "W_conv1": tf.Variable(tf.random_normal([3, 3, 3, 1, 64]), name="w_conv1"),
            #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
            "W_conv2": tf.Variable(tf.random_normal([3, 3, 3, 64, 128]), name="w_conv2"),
            #                                  64 features
            "W_fc": tf.Variable(tf.random_normal([4 * 4 * 64, 1000]), name="w_fc"),
            "out": tf.Variable(tf.random_normal([1000, n_classes]), name="out"),
        }

        biases = {
            "b_conv1": tf.Variable(tf.random_normal([32]), name="b_conv1"),
            "b_conv2": tf.Variable(tf.random_normal([64]), name="b_conv2"),
            "b_fc": tf.Variable(tf.random_normal([1000]), name="b_fc"),
            "out": tf.Variable(tf.random_normal([n_classes]), name="out"),
        }

        #                            image X      image Y        image Z           reshape the image to the correct size
        x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, IMG_SIZE_PX, 1])

        # ReLU = rectified linear units (negative values become 0)
        conv1 = tf.nn.relu(conv3d(x, weights["W_conv1"]) + biases["b_conv1"])
        conv1 = maxpool3d(conv1)

        conv2 = tf.nn.relu(conv3d(conv1, weights["W_conv2"]) + biases["b_conv2"])
        conv2 = maxpool3d(conv2)

        # final grid dimensions times the number of channel
        fc = tf.reshape(conv2, [-1, 4 * 4 * 64])
        fc = tf.nn.relu(tf.matmul(fc, weights["W_fc"]) + biases["b_fc"])
        fc = tf.nn.dropout(fc, keep_rate)

        output = tf.matmul(fc, weights["out"]) + biases["out"]

        return output
