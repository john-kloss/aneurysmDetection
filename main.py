from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import nn
import input
import preprocessing
from scipy.ndimage import rotate
import os


if __name__ == "__main__":
    # import the dicoms
    dicoms = input.import_dicoms()  
    
    # execute augmentation 
    # TODO: wrapper 

    #dicoms = preprocessing.augment.create_masks(dicoms)
    dicoms = preprocessing.augment.shear_images(dicoms)
    #dicoms = augment.rotate_images(dicoms)
    #dicoms = augment.scale_images(dicoms)
    #dicoms = augment.flip_images(dicoms)
    #dicoms = augment.normalize_grayscale_originals(dicoms)
    
    data = preprocessing.createSubvolumes.create_subvolumes(dicoms)
    
    nn.training.train_neural_network(data['train'], data['val'])
