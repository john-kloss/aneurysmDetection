from scipy.ndimage import rotate
import pydicom
import numpy as np
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

def plot(prediction, labels):
    dims = prediction.shape
    aneurysm_pred = np.where(prediction > 0.5)
    label = np.where(labels > 0.5)

    fig = plt.figure()
    ax = fig.add_subplot(121, projection="3d")
    ax.set_xlim([0, dims[0]])
    ax.set_ylim([0, dims[1]])
    ax.set_zlim([0, dims[2]])
    ax.scatter(
        aneurysm_pred[0],
        aneurysm_pred[1],
        aneurysm_pred[2],
        zdir="z",
        c="red",
        alpha=0.5,
        marker=".",
    )  

    ax.scatter(
        label[0],
        label[1],
        label[2],
        zdir="z",
        c="red",
        alpha=0.5,
        marker=".",
    )
    plt.savefig("fig.jpg") 

def create_masks(dicom):

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
                        mask[ac[0] - size + z][ac[1] - size + y][ac[2] - size + x] = 1

    dicom.mask = mask
    #plot(dicom.pixel_array, mask)   
    return dicom


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


def augmentation(dicom, num=9):


    pixel_array = []
    mask = []
    progressbar.printProgressBar(0, num, prefix="Creating Augmentations:", suffix="Complete", length=50)
    for n in range(num):
        tmp = rotate_images(dicom.pixel_array, dicom.mask)
        tmp = scale_images(tmp["pixel_array"], np.rint(tmp["mask"]))
        tmp = shear_images(tmp["pixel_array"], np.rint(tmp["mask"]))
        #p = [rotated["params"], scaled["params"], sheared["params"]]
        pixel_array.append(normalize_grayscale(tmp["pixel_array"]))
        mask.append(np.rint(tmp["mask"]))
        
        progressbar.printProgressBar(n+1, num, prefix="Creating Augmentations", suffix="Complete", length=50,)
    

    augmentations = {
        "pixel_array" : pixel_array,
        "mask" : mask, 
        "amount" : num
    }

    dicom.augmentations = augmentations
    
    return dicom

