
import input
import numpy as np
import os
import progressbar

SUBVOLUME_SIZE = 64
SUBVOLUME_OVERLAP = 0.5


def create_subvolumes(dicom):

    images = []
    labels = []


    dim = dicom.pixel_array.shape
    # calculate the stepsize
    stepsize = int(SUBVOLUME_SIZE * SUBVOLUME_OVERLAP)
    # initialize the progress bar


    progressbar.printProgressBar(
        0, dim[2], prefix="Creating Testset subvolumes:", suffix="Complete", length=50
    )
    # iterate over all dimension
    for x in range(0, dim[0] - SUBVOLUME_SIZE + 1, stepsize):
        for y in range(0, dim[1] - SUBVOLUME_SIZE + 1, stepsize):
            for z in range(0, dim[2] - SUBVOLUME_SIZE + 1, stepsize):
                # calculate the coordinates for the subvolume
                subvolume = dicom.pixel_array[
                    x : x + SUBVOLUME_SIZE,
                    y : y + SUBVOLUME_SIZE,
                    z : z + SUBVOLUME_SIZE,
                ]

                label = dicom.mask [
                    x : x + SUBVOLUME_SIZE,
                    y : y + SUBVOLUME_SIZE,
                    z : z + SUBVOLUME_SIZE,
                ]

                images.append(subvolume)
                labels.append(label)

                # print the progress
                progressbar.printProgressBar(
                    z + 1,
                    dim[2],
                    prefix="Creating Testset subvolumes",
                    suffix="Complete",
                    length=50,
                )


    

    # create the data object with train and test set
    data = {
        "images": images, 
        "labels": labels
    }
    return data


