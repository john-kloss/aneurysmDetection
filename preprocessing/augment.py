from scipy.ndimage import rotate
import pydicom
import numpy as np
import input
import pymrt as mrt
import pymrt.geometry
import transforms3d as trans
from scipy.ndimage import affine_transform
import random
import progressbar
import matplotlib.pyplot as plt


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




def rotate_images(pixel_array, mask):

    param = (np.random.randint(-15,15))
    pixel_array = rotate(pixel_array, axes=(2,1), angle=param, reshape=False) 
    mask = rotate(mask, axes=(0,1), angle=param, reshape=False)

    rotated_data = {
        "pixel_array" : pixel_array,
        "mask" : mask, 
        "params" : param
    }
    
    return rotated_data

def shear_images(pixel_array, mask):
    # only transform x axis?
    param = random.uniform(0, 0.05)
    S = [param, 0, 0]

    pixel_array = affine_transform(pixel_array,trans.shears.striu2mat(S))
    mask = affine_transform(mask,trans.shears.striu2mat(S))

    sheared_data = {
        "pixel_array" : pixel_array,
        "mask" : mask, 
        "params" : param
    }
    
    return sheared_data

def scale_images(pixel_array, mask):

    param = random.uniform(0.95,1.05)
    S = trans.zooms.zfdir2mat(param)
    pixel_array = affine_transform(pixel_array,S)
    mask = affine_transform(mask,S)
    
    scaled_data = {
        "pixel_array" : pixel_array,
        "mask" : mask, 
        "params" : param
    }
    
    return scaled_data


def flip_images(pixel_array, mask):
    # performs vertical flip, but not checked whether it fully works yet

    x,y,z = pixel_array.shape
    
    p = [int(x/2),int(y/2),int(z/2)]
    n = [0,0,-1]
    A = trans.reflections.rfnorm2aff(n,p)
    pixel_array = affine_transform(pixel_array,A)
    mask = affine_transform(mask,A)
    
    flipped_data = {
        "pixel_array" : pixel_array,
        "mask" : mask
    }

    return flipped_data


def augmentation(dicoms, num=9):
     # initialize the progress bar
    progressbar.printProgressBar(
        0, len(dicoms), prefix="Creating Augmentations:", suffix="Complete", length=50
    )

    for i in range(len(dicoms)):
        dicom = dicoms[i]

        pixel_array = []
        mask = []
        params = []
        
        for n in range(num):
            rotated = rotate_images(dicom.pixel_array, dicom.mask)
            scaled = scale_images(rotated["pixel_array"], np.rint(rotated["mask"]))
            sheared = shear_images(scaled["pixel_array"], np.rint(scaled["mask"]))
            p = [rotated["params"], scaled["params"], sheared["params"]]
            pixel_array.append(normalize_grayscale(sheared["pixel_array"]))
            mask.append(np.rint(sheared["mask"]))
            params.append(p)

        
        # print the progress
        progressbar.printProgressBar(i + 1, len(dicoms), prefix="Creating Augmentations", suffix="Complete", length=50,)

        augmentations = {
            "pixel_array" : pixel_array,
            "mask" : mask, 
            "params" : p,
            "amount" : num
        }

        dicom.augmentations = augmentations
    
    return dicoms

