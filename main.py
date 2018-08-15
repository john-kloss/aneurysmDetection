from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import nn
import input
from preprocessing.createSubvolumes import create_subvolumes
from preprocessing.augment import augmentation, create_masks 
from scipy.ndimage import rotate
import os
import numpy as npS


if __name__ == "__main__":
    # import the dicoms
    dicoms = input.import_dicoms()  
    
    
    dicoms = create_masks(dicoms)
    dicoms = augmentation(dicoms)

    #dicoms = augment.normalize_grayscale_originals(dicoms)
    
    data = create_subvolumes(dicoms)
    
    nn.training.train_neural_network(data['train'], data['val'])
