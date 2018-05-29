import binvox_rw
import os


def import_voxel():

    with open(os.getcwd() + '/data/Patient_C38_MRA_aneurysm.binvox', 'rb') as voxel:
        array = binvox_rw.read_as_3d_array(voxel)
        return array
