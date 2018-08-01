import input
import numpy as np
import binvox_rw
import os
import progressbar

SUBVOLUME_SIZE = 32
SUBVOLUME_OVERLAP = 0.2


def create_subvolumes(dicoms):

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    progressbar.printProgressBar(
        0, len(dicoms), prefix='Creating subvolumes:', suffix='Complete', length=50)

    for i in range(len(dicoms)):
        dicom = dicoms[i]
        dim = dicom.pixel_array.shape
        # calculate the stepsize
        stepsize = int(SUBVOLUME_SIZE*SUBVOLUME_OVERLAP)

        # iterate over all dimension
        for x in range(0, dim[0]-SUBVOLUME_SIZE+1, stepsize):
            for y in range(0, dim[1]-SUBVOLUME_SIZE+1, stepsize):
                for z in range(0, dim[2]-SUBVOLUME_SIZE+1, stepsize):
                    # calculate the coordinates for the subvolume
                    subvolume = dicom.pixel_array[x:x+SUBVOLUME_SIZE,
                                                  y:y+SUBVOLUME_SIZE,
                                                  z:z+SUBVOLUME_SIZE]

                    # check if the sv contains an aneurysm
                    label = check_for_aneurysm(x, y, z, dicom.aneurysm)

                    # add 80% of the sv to the training set
                    if (i < len(dicoms)*0.8):
                        train_images.append(subvolume)
                        train_labels.append(label)
                    else:  # add the rest to the test data
                        test_images.append(subvolume)
                        test_labels.append(label)

        # print the progress
        progressbar.printProgressBar(
            i+1, len(dicoms), prefix='Creating subvolumes', suffix='Complete', length=50)

    # create the data object with train and test set
    data = {'train': {'images': train_images, 'labels': train_labels},
            'test': {'images': test_images, 'labels': test_labels}}
    return data


# check if the subvolume contain the aneurysm
#

def check_for_aneurysm(x, y, z, aneurysm_coordinates):
    for i in range(len(aneurysm_coordinates)):
        ac = aneurysm_coordinates[i]
        if ((x < ac[0]-ac[3]) & (x+SUBVOLUME_SIZE > ac[0]+ac[3]) &
            (y < ac[1]-ac[3]) & (y+SUBVOLUME_SIZE > ac[1]+ac[3]) &
                (z < ac[2]-ac[3]) & (z+SUBVOLUME_SIZE > ac[2]+ac[3])):
            return True
    return False
