from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
from nn import training
import input
from preprocessing.createSubvolumes import create_subvolumes
from preprocessing.augment import augmentation, create_masks, normalize_grayscale
from preprocessing.dataStorage import init_storage, write_dicoms
from scipy.ndimage import rotate
import os
import h5py
import numpy as np
import progressbar


if __name__ == "__main__":
    # import the dicoms
    count = 0
    labs = None
    imgs = None
    hffile = None
    """
    for file in os.listdir(os.getcwd() + "/data"):
        if ".dcm" in file:
            dicom = input.import_dicom(file)
            print("Processing Patient "+ str(count+1))

            dicom = create_masks(dicom)
            dicom = augmentation(dicom,1)
            
            dicom.pixel_array = normalize_grayscale(dicom.pixel_array)
            dicom = create_subvolumes(dicom)
            
            
            init_storage(dicom,count)
            

            count += 1 
        
    """ 
    training.train_model()   

#    nn.training.train_neural_network(data['train'], data['val'])
