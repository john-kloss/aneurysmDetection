from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
from nn import training, prediction
import input
import preprocessing.createTestSetSubvolumes as testpp
import preprocessing.createSubvolumes as trainpp 
import preprocessing.augment as augment
from preprocessing.dataStorage import init_storage
import os
import numpy as np
import progressbar

#               0              1           2         3
ACTIONS = ['augment', 'create_testset', 'train', 'predict']
ACTION = ACTIONS[2] # <- select action index

if __name__ == "__main__":
    if ACTION == 'augment' or ACTION == 'create_testset':
        # import the dicoms
        count = 0
        labs = None
        imgs = None
        hffile = None
        for file in os.listdir(os.getcwd() + "/data/"):
            if ".dcm" in file:
                dicom = input.import_dicom(file)
                if len(dicom.aneurysm) == 0:
                    continue
                print("Processing Patient "+ str(count+1))

                dicom = augment.create_masks(dicom)
                
                if ACTION == 'create_testset':
                    # no augmentation step
                    dicom.pixel_array = augment.normalize_grayscale(dicom.pixel_array)
                    dicom = testpp.create_subvolumes(dicom)
                    init_storage(dicom,test=True)
                
                else: 
                    dicom = augment.augmentation(dicom)  
                    dicom.pixel_array = augment.normalize_grayscale(dicom.pixel_array)
                    dicom = trainpp.create_subvolumes(dicom)
                    init_storage(dicom)
                

                count += 1 
    elif ACTION == 'train':
    
        training.train_model()  

    elif ACTION == 'predict':
        evals = prediction.predict()

    else:
        raise Exception('Unknown action')
