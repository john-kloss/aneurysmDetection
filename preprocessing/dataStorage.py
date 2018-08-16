import os

import numpy as np
import tables
import h5py

def write_to_hdf5(data):
    
    train_images = data["train"]["images"]
    train_labels = data["train"]["labels"]
    val_labels = data["val"]["labels"]
    val_images = data["val"]["images"]

    data = h5py.File('./data/tmp/data2.h5', 'w')

    train = data.create_group("train")
    t_images = train.create_group("t_images")
    t_labels = train.create_group("t_labels")

    val = data.create_group("val")
    images = val.create_group("v_images")
    labels = val.create_group("v_labels")


    # train data 
    t_labels.create_dataset("train_labels", data=train_labels)
    t_images.create_dataset("train_images", data=train_images)

    # validation data
    labels.create_dataset("val_labels", data=val_labels)       
    images.create_dataset("val_images", data=val_images)

    data.close()

    

def access_data(path):
    file = h5py.File(path, 'r')
    g = file.get("train/t_labels")
    g["train_labels"].value
