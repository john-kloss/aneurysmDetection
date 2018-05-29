from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import input

if __name__ == "__main__":
    array = input.import_voxel()
    print(array.dims)
