from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
from nn import training, prediction
import input
from preprocessing.createSubvolumes import create_subvolumes
from preprocessing.augment import augmentation, create_masks, normalize_grayscale
from preprocessing.dataStorage import init_storage
import os
import h5py
import numpy as np
import progressbar

ACTIONS = ['augment', 'train', 'predict']
ACTION = ACTIONS[0] # <- select action index

if __name__ == "__main__":
    if ACTION == 'augment':
        # import the dicoms
        count = 0
        labs = None
        imgs = None
        hffile = None
        for file in os.listdir(os.getcwd() + "/data"):
            if ".dcm" in file:
                dicom = input.import_dicom(file)
                if len(dicom.aneurysm) == 0:
                    continue
                print("Processing Patient "+ str(count+1))

                dicom = create_masks(dicom)

                dicom = augmentation(dicom)
                
                dicom.pixel_array = normalize_grayscale(dicom.pixel_array)
                dicom = create_subvolumes(dicom)
                
                
                init_storage(dicom)
                

                count += 1 
    elif ACTION == 'train':
    
        training.train_model()   
    elif ACTION == 'predict':
        evals = prediction.predict()

    else:
        raise Exception('Unknown action')
