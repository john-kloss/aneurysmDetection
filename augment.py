from scipy.ndimage import rotate
import pydicom
import numpy as np
import inputs
import pymrt as mrt
import pymrt.geometry
import transforms3d as trans
from scipy.ndimage import affine_transform
import random

NUM_ROTATE = 1
NUM_SHEAR = 1
NUM_SCALES = 1 
NUM_FLIPS = 1

def normalize_grayscale(dicoms):
    for dicom in dicoms:
        min = np.min(dicom.pixel_array)
        max = np.max(dicom.pixel_array)
        dicom.pixel_array = (dicom.pixel_array - min) / (max - min)


def create_masks(dicoms):
        # iterate over all dicoms
    for i in range(len(dicoms)):
        dicom = dicoms[i]

         # generate mask
        mask = np.zeros((dicom.pixel_array.shape))

        for ac in dicom.aneurysm: 
            # make shape of sphere array odd so centroid is exactly one voxel
            size = int(ac[3])
            shape = size*2+1 if size%2==0 else size*2
            aneurysm_sphere = mrt.geometry.sphere(shape,size)

            # create mask by laying aneurysm sphere over dicom pixelarray 
            for x in range(shape):
                for y in range(shape):
                    for z in range(shape):
                        # set all voxels containing aneurysm to 1 
                        if aneurysm_sphere[x][y][z]:
                            mask[ac[0] - size + x][ac[1] - size + y][ac[2] - size + z] = 1

        dicom.mask = mask



def rotate_images(dicoms):
    for dicom in dicoms:
        pixel_array = []
        mask = []
        params = []

        for n in range(NUM_ROTATE):
            param = (np.random.randint(1,175))
            pixel_array.append(rotate(dicom.pixel_array, axes=(0,1), angle=param, reshape=False))
            mask.append(rotate(dicom.mask, axes=(0,1), angle=param, reshape=False))
            params.append(param)

        rotated_data = {
            "pixel_array" : pixel_array,
            "mask" : mask, 
            "params" : params
        }

        dicom.rotations = rotated_data

def shear_images(dicoms):
    for dicom in dicoms:
        pixel_array = []
        mask = []
        params = []
        for shears in range(NUM_SHEAR):
            # only transform x axis?
            param = random.uniform(0.5, 3.5)
            S = [param, 0, 0]
            pixel_array.append(affine_transform(dicom.pixel_array,trans.shears.striu2mat(S)))
            mask.append(affine_transform(dicom.mask,trans.shears.striu2mat(S)))
            params.append(param)

        sheared_data = {
            "pixel_array" : pixel_array,
            "mask" : mask, 
            "params" : params

        }

        dicom.shears = sheared_data

def scale_images(dicoms):
    for dicom in dicoms:
        pixel_array = []
        mask = []
        params = []
        for scales in range(NUM_SCALES):
            param = random.uniform(0.5,2)
            S = trans.zooms.zfdir2mat(param)
            pixel_array.append(affine_transform(dicom.pixel_array,S))
            mask.append(affine_transform(dicom.mask,S))
            params.append(param)
        
        scaled_data = {
            "pixel_array" : pixel_array,
            "mask" : mask, 
            "params" : params
        }

        dicom.scales = scaled_data


def flip_images(dicoms):
    for dicom in dicoms:
        pixel_array = []
        mask = []
        params = []

        x,y,z = dicom.pixel_array.shape
        for flips in range(NUM_SCALES):
            p = [int(x/2),int(y/2),int(z/2)]
            n = [0,0,-1]
            A = trans.reflections.rfnorm2aff(n,p)
            pixel_array.append(affine_transform(dicom.pixel_array,A))
            mask.append(affine_transform(dicom.mask,A))
        
        flipped_data = {
            "pixel_array" : pixel_array,
            "mask" : mask, 
            "params" : params
        }

        dicom.flips = flipped_data