from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import nn
import inputs
import createSubvolume
import augment
from scipy.ndimage import rotate
import os


if __name__ == "__main__":
    # import the dicoms
    dicoms = inputs.import_dicoms()  

    #augment.normalize_grayscale(dicoms)

    #augment.create_masks(dicoms)
    #augment.shear_images(dicoms)
    #augment.rotate_images(dicoms)
    
    #augment.scale_images(dicoms)
    augment.flip_images(dicoms)

   # nn.train_neural_network(data['train'], data['val'])
