from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import input
import conv3dnet
x = tf.placeholder('float')


if __name__ == "__main__":
    array = input.import_voxel()
    print(array.dims)
    conv3dnet.train_neural_network(x, array.data, array.data)
