from scipy.ndimage import rotate
import pydicom
import numpy as np
import input
import pymrt as mrt
import pymrt.geometry
import transforms3d as trans
from scipy.ndimage import affine_transform
import random
import matplotlib.pyplot as plt
from collections import Counter



def normalize_grayscale(pixel_array):
    min = np.min(pixel_array)
    max = np.max(pixel_array)
    normalized_array = (pixel_array - min) / (max - min)
    
    return normalized_array


def normalize_grayscale_originals(dicoms):
    for dicom in dicoms:
        min = np.min(dicom.pixel_array)
        max = np.max(dicom.pixel_array)
        dicom.pixel_array = (dicom.pixel_array - min) / (max - min)
    
    return dicoms

def create_masks(dicoms):
        # iterate over all dicoms
    for i in range(len(dicoms)):
        dicom = dicoms[i]

         # generate mask
        mask = np.zeros((dicom.pixel_array.shape))

        for ac in dicom.aneurysm: 
            # make shape of sphere array odd so centroid is exactly one voxel
            size = int(ac[3])  # size equals radius b/c size given in mm, one voxel appr. 0,5 mm 
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
    
    return dicoms




def rotate_images(dicom, num):
    pixel_array = []
    mask = []
    params = []

    for n in range(num):
        param = (np.random.randint(1,30))
        rotation = rotate(dicom.pixel_array, axes=(2,1), angle=param, reshape=False)
        
        pixel_array.append(rotation)
        mask.append(rotate(dicom.mask, axes=(0,1), angle=param, reshape=False))
        params.append(param)

    rotated_data = {
        "pixel_array" : pixel_array,
        "mask" : mask, 
        "params" : params
    }

    dicom.rotations = rotated_data
    
    return dicom

def shear_images(dicom, num):
    pixel_array = []
    mask = []
    params = []
    for shears in range(num):
        # only transform x axis?
        # param = random.uniform(0.5, 3.5)
        param = 2
        S = [param, 0, 0]
        shear = affine_transform(dicom.pixel_array,trans.shears.striu2mat(S))
        
        pixel_array.append(shear)
        #mask.append(affine_transform(dicom.mask,trans.shears.striu2mat(S)))
        params.append(param)

    sheared_data = {
        "pixel_array" : pixel_array,
        "mask" : mask, 
        "params" : params

    }

    dicom.shears = sheared_data
    
    return dicom

def scale_images(dicom, num):
    
    pixel_array = []
    mask = []
    params = []
    for scales in range(num):
        #param = random.uniform(0.5,2)
        param = 1.5
        S = trans.zooms.zfdir2mat(param)
        scaling = affine_transform(dicom,S)
        
        pixel_array.append(scaling)
        #mask.append(affine_transform(dicom.mask,S))
        params.append(param)
    
    scaled_data = {
        "pixel_array" : pixel_array,
        "mask" : mask, 
        "params" : params
    }

    #dicom.scales = scaled_data
    
    return scaling


def flip_images(dicom, num):

    pixel_array = []
    mask = []
    params = []

    x,y,z = dicom.pixel_array.shape
    for flips in range(num):
        p = [int(x/2),int(y/2),int(z/2)]
        n = [0,0,-1]
        A = trans.reflections.rfnorm2aff(n,p)
        flip = affine_transform(dicom.pixel_array,A)

        pixel_array.append(normalize_grayscale(flip))
        mask.append(affine_transform(dicom.mask,A))
    
    flipped_data = {
        "pixel_array" : pixel_array,
        "mask" : mask, 
        "params" : params
    }

    dicom.flips = flipped_data
    
    return dicom


def augmentation(dicoms):
    # combination of 9 augmentations
    augment_types= random.choices(["shear_images", "rotate_images", "scale_images"], k=9)
    counts = Counter(augment_types)

    #
    # for augmentation in dicoms:

    #dicoms = create_masks(dicoms)
    #dicoms = shear_images(dicoms)
    #dicoms = rotate_images(dicoms)
    #dicoms = scale_images(dicoms)