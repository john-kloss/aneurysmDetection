import h5py
import os
import numpy as np
import keras
"""




class DataGenerator(keras.utils.Sequence):

    def __init__(self,batch_size,x_set,y_set):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        res = np.empty()
        for i in batch_x:
            f = h5py.File(os.getcwd() + "/data/processed/new/dicom0.h5", 'r')
            print(i)

            return np.array(np.resize(f['images/images'][i], (1,64,64,64))), np.array(np.resize(f['labels/labels'][i], (1,64,64,64)))

"""

def data_gen(patients,batch_size, epoch_steps):
    """
    Generator to yield inputs and their labels in batches.
    """
    
    idx = 0

    while True:
        for pat in patients:
            if idx>10:
                idx = 0
        

            X = np.empty((batch_size,1,64,64,64))
            y = np.empty((batch_size,1,64,64,64))
            for i in range(batch_size):

                f = h5py.File(os.getcwd() + "/data/processed/dicom"+str(pat)+".h5", 'r')
                X[i,] = np.resize(f['images']['images'][idx+i], (1,64,64,64))

                y[i,] = np.resize(f['labels']['labels'][idx+i], (1,64,64,64))
            print(" processing sample "+str(idx))
            
            idx += 1
            yield X,y


