
import input
import numpy as np
import os
import progressbar

SUBVOLUME_SIZE = 64
SUBVOLUME_OVERLAP = 1
IMAGE_CENTER = 0.8


def create_subvolumes(dicom):
    """
    create subvolumes for testing model - one of the subvolumes fully contains aneurysm 
    """

    images = []
    labels = []
    cuts = []

    dim = dicom.pixel_array.shape
    # calculate the stepsize
    stepsize = int(SUBVOLUME_SIZE * SUBVOLUME_OVERLAP)
    # initialize the progress bar

    
    progressbar.printProgressBar(
        0, dim[2], prefix="Creating Testset subvolumes:", suffix="Complete", length=50
    )
    # get dimension ranges as center as 80% of image
    X_dim_start = (dim[0]*(1-IMAGE_CENTER))/2
    Y_dim_start = (dim[1]*(1-IMAGE_CENTER))/2
    Z_dim_start = (dim[2]*(1-IMAGE_CENTER))/2

    # get corner vertex of subvolume containing aneurysm 
    x = dicom.aneurysm[0][0]-32
    y = dicom.aneurysm[0][1]-32
    z = dicom.aneurysm[0][2]-32

    # compute starting corner of 80% central image from which starting to cut would result in whole aneurysm included in image

    # get distance between dimension range and aneurysm corner vertex and compute how many times a subvolume would fit, take the rounded 
    # value times subvolume size to subtract from corner vertex to get starting corner vertex  
    X_START = int(max(0, x - int((x-X_dim_start)/64) *64))
    Y_START = int(max(0,y - int((y-Y_dim_start)/64) *64))
    Z_START = int(max(0,z - int((z-Z_dim_start)/64) *64))

    # start with aneurysm in center
    for x in range(X_START, dim[0] - SUBVOLUME_SIZE + 1, stepsize):
        for y in range(Y_START, dim[1] - SUBVOLUME_SIZE + 1, stepsize):
            for z in range(Z_START, dim[2] - SUBVOLUME_SIZE + 1, stepsize):
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

                cuts.append([x+32,y+32,z+32])
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


    
    data = {
        "images": images, 
        "labels": labels,
        "patient": dicom.patient,
        "coordinates": cuts,
        "pixel_array": dicom.pixel_array
    }
    return data


