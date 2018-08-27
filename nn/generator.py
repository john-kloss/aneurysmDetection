import h5py
import os
import numpy as np
import keras

import re
path = os.getcwd() + "/data/processed" 

class DataGenerator(keras.utils.Sequence):

    def __init__(self,batch_size,x_set):
        self.x = x_set
        self.batch_size = batch_size

    def __len__(self):
        return int(960*len(self.x))

    def __getitem__(self, idx):
        for filename in os.listdir(path):
            if re.match(".*#" + str(int(np.floor(idx/960))) + "#.*", filename):
                f = h5py.File( path +'/' + filename , 'r')
                return np.array(np.resize(f['images/images'][idx%960], (1,1,64,64,64))), np.array(np.resize(f['labels/labels'][idx%960], (1,1,64,64,64)))
