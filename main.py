from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import input
import nn
import createSubvolume
import augment
from scipy.ndimage import rotate
import os

if __name__ == "__main__":
    # import the dicoms
    dicoms = input.import_dicoms()
    #ds=dicoms[1]
    #rot = rotate(ds.pixel_array, angle=90,axes=(0,2), reshape=False)
    #ds.PixelData=rot
    #ds.save_as(os.getcwd()+"/data/tmp/test_02.dcm")

    #augmented_dicoms = augment.rotate_image(dicoms)
    data = createSubvolume.create_subvolumes(dicoms)
    nn.train_neural_network(data['train'], data['val'])
