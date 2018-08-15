import os

import numpy as np
import tables
import h5py

data = {
        "train": {"images": np.arange(64*64*64).reshape(64,64,64), "labels": np.arange(64*64*64).reshape(64,64,64)},
        "val" : {"images": np.arange(64*64*64).reshape(64,64,64), "labels": np.arange(64*64*64).reshape(64,64,64)},
        "test": {"images": np.arange(64*64*64).reshape(64,64,64), "labels": np.arange(64*64*64).reshape(64,64,64)}
    }

train_images = data["train"]["images"]
train_labels = data["train"]["labels"]
val_labels = data["val"]["labels"]
val_labels = data["val"]["images"]
    
data = h5py.File('data.h5', 'w')
train = data.create_group("train")
images = train.create_group("images")
labels = train.create_group("labels")

labels.create_dataset("labels", train_labels)
images.create_dataset("images", train_images)


data.close()

hf = h5py.File('data.h5', 'r')
hf.keys()
#print(hf.keys)