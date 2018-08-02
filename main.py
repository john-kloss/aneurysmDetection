from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import input
import numpy as np
import nn
import createSubvolumes

if __name__ == "__main__":
    # import the dicoms
    dicoms = input.import_dicoms()
    data = createSubvolumes.create_subvolumes(dicoms)

    nn.train_neural_network(data['train'], data['test'])
