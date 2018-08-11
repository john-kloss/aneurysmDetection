import input
import numpy as np
import os
import progressbar
import math
import scipy.stats as stats
import pymrt as mrt
import pymrt.geometry


SUBVOLUME_AMOUNT=10000
SUBVOLUME_SIZE = 32
#SUBVOLUME_OVERLAP = 0.5



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


        # generate mask
        mask = np.zeros((dicom.pixel_array.shape))

        for ac in dicom.aneurysm: 
            # make shape of sphere array odd so centroid is exactly one voxel
            size = int(ac[3])
            shape = size*2+1 if size%2==0 else size*2
            aneurysm_sphere = mrt.geometry.sphere(shape,size)

            #
            for x in range(shape):
                for y in range(shape):
                    for z in range(shape):
                        # set all voxels containing aneurysm to 1 
                        if aneurysm_sphere[x][y][z]:
                            mask[ac[0] - size + x][ac[1] - size + y][ac[2] - size + z] = 1


        sv = int(SUBVOLUME_SIZE/2)
        for num_aneurysm in range(len(dicom.aneurysm)):
            for n in range(SUBVOLUME_AMOUNT):                
                # draw random numbers from normal distribution for each dimension around aneurysm coordinate
                sig = 40
                sv_centroid = [
                    int (stats.truncnorm.rvs( 
                        a = (sv - dicom.aneurysm[num_aneurysm][coords]) / sig , 
                        b = (dim[coords]-sv -  dicom.aneurysm[num_aneurysm][coords]) / sig, 
                        loc = dicom.aneurysm[num_aneurysm][coords], 
                        scale = sig) ) 
                        
                for coords in range(3) ]
                
                
                subvolume = dicom.pixel_array [
                    (sv_centroid[0] - sv) : (sv_centroid[0] + sv),
                    (sv_centroid[1] - sv) : (sv_centroid[1] + sv),
                    (sv_centroid[2] - sv) : (sv_centroid[2] + sv)
                ]

                label = mask [
                    (sv_centroid[0] - sv) : (sv_centroid[0] + sv),
                    (sv_centroid[1] - sv) : (sv_centroid[1] + sv),
                    (sv_centroid[2] - sv) : (sv_centroid[2] + sv)
                ]

                label_true_false = [0,1] if ((label >0).sum()) == 0 else [1,0]




                # add 80% of the subvolume to the training set
                if i < len(dicoms) * 0.8:
                    train_images.append(subvolume)
                    train_labels.append(label_true_false)
                elif i < len(dicom) * 0.9:
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
        "train": {"images": train_images[0:100], "labels": train_labels[0:100]},
        "val" : {"images": val_images[0:100], "labels": val_labels[0:100]},
        "test": {"images": test_images[0:100], "labels": test_labels[0:100]}
    }
    return data






