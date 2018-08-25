import os

import numpy as np
import tables
import h5py

    
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

    dt = "(64,64,64)f8"
    labs = labels.create_dataset("labels", (len(dat["labels"]),), maxshape=(None,), chunks=True, dtype=dt)
    imgs = images.create_dataset("images", (len(dat["images"]),), maxshape=(None,), chunks=True, dtype=dt)
    imgs[:] = dat["images"]
    labs[:] = dat["labels"]

    data.close()


def access_data(path):
    file = h5py.File(path, 'r')
    g=file['images']['images'].va
    g["labels"].value

#access_data('./data/processed/new/dicom0.h5')
