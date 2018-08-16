
# Imports
import nn
import input
import preprocessing
from scipy.ndimage import rotate
import os


if __name__ == "__main__":
<<<<<<< HEAD

        # import the dicoms
        dicoms = inputs.import_dicoms()  
        augment.create_masks(dicoms)
        data = createSubvolumes.create_subvolumes(dicoms)
        # execute augmentation
        augment.shear_images(dicoms)
        augment.rotate_images(dicoms)
        augment.scale_images(dicoms)
        #augment.flip_images(dicoms)
        
        augment.normalize_grayscale(dicoms)

        nn.train_neural_network(data['train'], data['val'])
=======
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
>>>>>>> 735e2de62ccaa9c270f25835c44fb1dcd618ccbd
