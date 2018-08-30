import os

import numpy as np
import tables
import h5py


def init_storage(dat, test=False):
    path = os.getcwd() + "/data/processed/processed_range20/" 

    if test:
        path = os.getcwd() + "/data/processed/test/" 

    patient_number = len(os.listdir(path))
    data = h5py.File(path+str(dat["patient"])+ '#' + str(patient_number) + '#' +'.h5', 'w')
    images = data.create_group("images")
    labels = data.create_group("labels")

    dt = "(64,64,64)f8"
    labs = labels.create_dataset("labels", (len(dat["labels"]),), maxshape=(None,), chunks=True, dtype=dt)
    imgs = images.create_dataset("images", (len(dat["images"]),), maxshape=(None,), chunks=True, dtype=dt)
    imgs[:] = dat["images"]
    labs[:] = dat["labels"]

    data.close()
