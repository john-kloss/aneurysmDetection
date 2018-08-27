import os

import numpy as np
import tables
import h5py

path = os.getcwd() + "/data/processed" 


def init_storage(dat):
    patient_number = len(os.listdir(path))
    data = h5py.File('./data/processed/'+str(dat["patient"])+ '#' + str(patient_number) + '#' +'.h5', 'w')
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
