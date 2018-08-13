
# Imports
import input
import nn
import createSubvolumes
import tensorflow as tf

if __name__ == "__main__":
    with tf.device('/gpu:0'):

        #import the dicoms
        dicoms = input.import_dicoms()
        data = createSubvolumes.create_subvolumes(dicoms)

        nn.train_neural_network(data['train'], data['test'])

