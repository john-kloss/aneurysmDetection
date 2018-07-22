import input
import numpy as np

import binvox_rw
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SUBVOLUME_FACTOR = 2
SUBVOLUME_OVERLAP = 0.2
aneurysm_coordinates = [80, 135, 135, 175, 100, 150]


def create_subvolumes():
    voxel = input.import_single_voxel('Patient_14_MRA.binvox')
    dim = voxel.data.shape[0]
    # calculate the dimension for the subvolume
    dim_sv = int(dim/SUBVOLUME_FACTOR)
    stepsize = int(dim_sv*SUBVOLUME_OVERLAP)
    subvolumes = []
    labels = []
    # iterate over all dimension
    for x in range(0, dim-dim_sv+1, stepsize):
        for y in range(0, dim-dim_sv+1, stepsize):
            for z in range(0, dim-dim_sv+1, stepsize):
                # calculate the coordinates for the subvolume
                subvolume = voxel[x:x+dim_sv,
                                  y:y+dim_sv,
                                  z:z+dim_sv]
                # add the sv to the training data
                subvolumes.append(subvolume)
                # check if the sv contains an aneurysm
                label = check_for_aneurysm(
                    [x, x+dim_sv,
                     y, y+dim_sv,
                     z, z+dim_sv], aneurysm_coordinates)
                labels.append(label)

    print(labels)


# check if the subvolume contain the aneurysm

def check_for_aneurysm(sv_coordinates, aneurysm_coordinates):
    if ((sv_coordinates[0] < aneurysm_coordinates[0]) & (sv_coordinates[1] > aneurysm_coordinates[1]) &
        (sv_coordinates[2] < aneurysm_coordinates[2]) & (sv_coordinates[3] > aneurysm_coordinates[3]) &
            (sv_coordinates[4] < aneurysm_coordinates[4]) & (sv_coordinates[5] > aneurysm_coordinates[5])):
        return True
    return False
