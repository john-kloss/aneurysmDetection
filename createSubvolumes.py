import input
import numpy as np

import binvox_rw
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SUBVOLUME_FACTOR = 8
SUBVOLUME_OVERLAP = 0.5

if __name__ == "__main__":

    voxel = input.import_single_voxel('Patient_14_MRA.binvox')
    dim = voxel.data.shape[0]
    # calculate the dimension for the subvolume
    dim_sv = int(dim/SUBVOLUME_FACTOR)
    subvolumes = []
    labels = []
    # iterate over all dimension
    for x in range(0, dim, int(dim_sv*SUBVOLUME_OVERLAP)):
        for y in range(0, dim, int(dim_sv*SUBVOLUME_OVERLAP)):
            for z in range(0, dim, int(dim_sv*SUBVOLUME_OVERLAP)):
                subvolume = voxel[x:x+dim_sv,
                                  y:y+dim_sv,
                                  z:z+dim_sv]
                subvolumes.append(subvolume)
                # TODO: get correct label
                labels.append(False)
