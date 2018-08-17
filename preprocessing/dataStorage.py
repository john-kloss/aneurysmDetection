import os

import numpy as np
import tables
import h5py

def write_to_hdf5(data):

    images = data["images"]
    labels = data["labels"]

    data = h5py.File('./data/tmp/data_1608.h5', 'w')

    images = data.create_group("images")
    labels = data.create_group("labels")

    labels.create_dataset("labels", data=labels)
    images.create_dataset("images", data=images)

    data.close()

    
def write_dicoms(data, labs, imgs, file):


    imgs.resize(imgs.shape[0] + len(data["images"]), axis= 0)
    labs.resize(labs.shape[0] + len(data["labels"]), axis= 0)
    labs[-len(data["labels"]):] = data["labels"]
    imgs[-len(data["images"]):] = data["images"]
    file.close()



def init_storage(dat,i):
    data = h5py.File('./data/tmp/dicom'+str(i)+'.h5', 'w')
    images = data.create_group("images")
    labels = data.create_group("labels")

    dt_im = "(64,64,64)f8"
    dt_lab = "(2,)f8"
    labs = labels.create_dataset("labels", (len(dat["labels"]),), maxshape=(None,), chunks=True, dtype=dt_lab)
    imgs = images.create_dataset("images", (len(dat["images"]),), maxshape=(None,), chunks=True, dtype=dt_im)
    imgs[:] = dat["images"]
    labs[:] = dat["labels"]

    return labs, imgs, data



def access_data(path):
    file = h5py.File(path, 'r')
    g = file.get("/t_labels")
    g["labels"].value

#access_data('./data/tmp/dicoms.h5')