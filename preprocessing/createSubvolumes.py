import numpy as np
import os
import progressbar
import math
import scipy.stats as stats
from .augment import normalize_grayscale
import matplotlib.pyplot as plt



SUBVOLUME_AMOUNT = 80
SUBVOLUME_SIZE = 64

#ANEURYSM_COVERAGE = 0.95

def create_subvolumes(dicom, slack = 3):

    images = []
    labels = []


    # initialize the progress bar
    progressbar.printProgressBar(
        0, dicom.augmentations["amount"]+slack, prefix="Creating subvolumes:", suffix="Complete", length=50
    )

   
    dim = dicom.pixel_array.shape
    SUBVOLUMES = int(SUBVOLUME_AMOUNT / len(dicom.aneurysm))

    # define slack to generate more subvolumes from original data than augmentations
    
    for num_augments in range(dicom.augmentations["amount"]+slack):
        if num_augments < slack: 
            pixel_array = dicom.pixel_array
            mask = dicom.mask
        else:
            pixel_array = dicom.augmentations["pixel_array"][num_augments-slack]
            mask = dicom.augmentations["mask"][num_augments-slack]


        sv = int(SUBVOLUME_SIZE/2)
        for num_aneurysm in range(len(dicom.aneurysm)):    
            for n in range(SUBVOLUMES): 
                            
                # draw random numbers from normal distribution for each dimension around aneurysm coordinate
                # augmentations changed only slightly so aneurysm should still be in this range
                sig = 20
                sv_centroid = [
                    int (stats.truncnorm.rvs( 
                        a = (sv - dicom.aneurysm[num_aneurysm][coords]) / sig , 
                        b = ( (dim[coords]-sv ) -  dicom.aneurysm[num_aneurysm][coords]) / sig, 
                        loc = dicom.aneurysm[num_aneurysm][coords], 
                        scale = sig) ) 
                        
                for coords in range(3) ]
                
                
                subvolume = pixel_array[
                    (sv_centroid[0] - sv) : (sv_centroid[0] + sv),
                    (sv_centroid[1] - sv) : (sv_centroid[1] + sv),
                    (sv_centroid[2] - sv) : (sv_centroid[2] + sv)
                ]

                label = mask[
                    (sv_centroid[0] - sv) : (sv_centroid[0] + sv),
                    (sv_centroid[1] - sv) : (sv_centroid[1] + sv),
                    (sv_centroid[2] - sv) : (sv_centroid[2] + sv)
                ]

                # assign label to subvolume if aneurysm is covered to a percentage 
                #aneurysm_fraction = dicom.aneurysm[num_aneurysm][3]/sum( i[3] for i in dicom.aneurysm)

                #expected_coverage = len(mask.nonzero()[0]) * ANEURYSM_COVERAGE * aneurysm_fraction 

                # [1,0] true [0,1] false
                #label_true_false = np.array([0,1]) if (len(label.nonzero()[0]) <= expected_coverage) else np.array([1,0])

                images.append(subvolume)
                labels.append(label)
                    

        # print the progress
        progressbar.printProgressBar(
            num_augments + 1,
            dicom.augmentations["amount"]+slack,
            prefix="Creating subvolumes",
            suffix="Complete",
            length=50,
        )

    data = {
        "patient": dicom.patient,
        "images": images, 
        "labels": labels
    }
    return data






