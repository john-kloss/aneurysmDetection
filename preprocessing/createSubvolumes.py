import input
import numpy as np
import os
import progressbar
import math
import augment
import scipy.stats as stats


<<<<<<< HEAD:createSubvolumes.py
SUBVOLUME_AMOUNT=5000
SUBVOLUME_SIZE = 32
=======
SUBVOLUME_AMOUNT = 10
SUBVOLUME_SIZE = 64
ANEURYSM_COVERAGE = 0.95
>>>>>>> 735e2de62ccaa9c270f25835c44fb1dcd618ccbd:preprocessing/createSubvolumes.py

def create_subvolumes(dicoms):

    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    test_images = []
    test_labels = []

    # initialize the progress bar
    progressbar.printProgressBar(
        0, len(dicoms), prefix="Creating subvolumes:", suffix="Complete", length=50
    )

    # iterate over all dicoms
    for i in range(len(dicoms)):
        dicom = dicoms[i]
        dim = dicom.pixel_array.shape

        sv = int(SUBVOLUME_SIZE/2)
        for n in range(SUBVOLUME_AMOUNT): 
            for num_aneurysm in range(len(dicom.aneurysm)):               
                # draw random numbers from normal distribution for each dimension around aneurysm coordinate
                sig = 40
                sv_centroid = [
                    int (stats.truncnorm.rvs( 
                        a = (sv - dicom.aneurysm[num_aneurysm][coords]) / sig , 
                        b = ( (dim[coords]-sv ) -  dicom.aneurysm[num_aneurysm][coords]) / sig, 
                        loc = dicom.aneurysm[num_aneurysm][coords], 
                        scale = sig) ) 
                        
                for coords in range(3) ]
                
                
                subvolume = dicom.pixel_array [
                    (sv_centroid[0] - sv) : (sv_centroid[0] + sv),
                    (sv_centroid[1] - sv) : (sv_centroid[1] + sv),
                    (sv_centroid[2] - sv) : (sv_centroid[2] + sv)
                ]

                label = dicom.mask [
                    (sv_centroid[0] - sv) : (sv_centroid[0] + sv),
                    (sv_centroid[1] - sv) : (sv_centroid[1] + sv),
                    (sv_centroid[2] - sv) : (sv_centroid[2] + sv)
                ]

                # assign label to subvolume if aneurysm is covered to a percentage 
                aneurysm_fraction = dicom.aneurysm[num_aneurysm][3]/sum( i[3] for i in dicom.aneurysm)
                expected_coverage = len(dicom.mask.nonzero()[0]) * ANEURYSM_COVERAGE * aneurysm_fraction 

                # [1,0] true [0,1] false
                label_true_false = np.array([0,1]) if (len(label.nonzero()[0]) <= expected_coverage) else np.array([1,0])


                # add 80% of the patients to the training set
                if i == 0:#  len(dicoms) * 0.8:
                    train_images.append(subvolume)
                    train_labels.append(label_true_false)
                elif i ==1: # < len(dicom) * 0.9:
                    val_images.append(subvolume)
                    val_labels.append(label_true_false)
                else:  # add the rest to the test data
                    test_images.append(subvolume)
                    test_labels.append(label_true_false)

        # print the progress
        progressbar.printProgressBar(
            i + 1,
            len(dicoms),
            prefix="Creating subvolumes",
            suffix="Complete",
            length=50,
        )

    # create the data object with train and test set
    data = {
        "train": {"images": train_images[0:1000], "labels": train_labels[0:1000]},
        "val" : {"images": val_images[0:1000], "labels": val_labels[0:1000]},
        "test": {"images": test_images[0:1000], "labels": test_labels[0:1000]}
    }
    return data






