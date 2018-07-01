from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import input
import numpy as np
import nn

'''
data : {
    test: {
        images: [],
        labels: []
    },
    train: {
        images: [],
        labels: []
    }
}
'''


if __name__ == "__main__":
    data = input.import_voxel()
    nn.train_neural_network(data.train, data.test)
