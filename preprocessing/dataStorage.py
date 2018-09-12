import os

import numpy as np
import tables
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def init_storage(dat, test=False):
    path = os.getcwd() + "/data/processed/" 

    if test:
        path = os.getcwd() + "/data/processed/test/" 

    patient_number = len(os.listdir(path))+25
    data = h5py.File(path+str(dat["patient"])+ '#' + str(patient_number) + '#' +'.h5', 'w')
    images = data.create_group("images")
    labels = data.create_group("labels")
    

    dt = "(64,64,64)f8"
    labs = labels.create_dataset("labels", (len(dat["labels"]),), maxshape=(None,), chunks=True, dtype=dt)
    imgs = images.create_dataset("images", (len(dat["images"]),), maxshape=(None,), chunks=True, dtype=dt)
    imgs[:] = dat["images"]
    labs[:] = dat["labels"]

    if test: 
        
        cuts = data.create_group("cuts")
        coords = cuts.create_dataset("cuts",data=np.array(dat["coordinates"]).reshape(len(dat["coordinates"]),3))
        pixel = data.create_group("pixel")
        pix = pixel.create_dataset("pixel", data=dat["pixel_array"])

    data.close()

"""

f = h5py.File(os.getcwd() + "/data/processed/Patient_16_MRA#2#.h5",'r' )

amount_of_subvolumes = len(f["images/images"])
for i in range(amount_of_subvolumes):

    labels = np.array(np.reshape(f["labels/labels"][i], (1, 1, 64, 64, 64)))
    images = np.array(np.reshape(f["images/images"][i], (1, 1, 64, 64, 64)))
    images = np.where(images[0][0]>0.5)

    labels = np.where(labels[0][0]>0.9)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim([0, 64])
    ax.set_ylim([0, 64])
    ax.set_zlim([0, 64])
    
    ax.scatter(
        labels[0],
        labels[1],
        labels[2],
        zdir="z",
        c="r",
        alpha=0.05,
        marker=",",
        )
    ax.scatter(
        images[0],
        images[1],
        images[2],
        zdir="y",
        c= "0.4",
        alpha=0.5,
        marker=",",
        )
    
  
    
    plt.show()
"""
