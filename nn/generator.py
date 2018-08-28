import h5py
import os
import numpy as np
import keras

import re
path = os.getcwd() + "/data/processed" 
AMOUNT_SUBVOLUMES = 960

class DataGenerator(keras.utils.Sequence):

    def __init__(self,x_set):
        self.x = x_set

    def __len__(self):
        return int(AMOUNT_SUBVOLUMES*len(self.x))

    def __getitem__(self, idx):
        for filename in os.listdir(path):
            if re.match(".*#" + str(int(np.floor(idx/AMOUNT_SUBVOLUMES))) + "#.*", filename):
                f = h5py.File( path +'/' + filename , 'r')
                return np.array(np.resize(f['images/images'][idx%AMOUNT_SUBVOLUMES], (1,1,64,64,64))), np.array(np.resize(f['labels/labels'][idx%AMOUNT_SUBVOLUMES], (1,1,64,64,64)))
