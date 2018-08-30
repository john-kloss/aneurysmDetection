import os
import numpy as np
import tables
import h5py
from .model import unet_model_3d
from .generator import DataGenerator 
from .metrics import dice_coefficient
from scipy.spatial import distance
import matplotlib.pyplot as plt

def visualize_mask(mask):
    plt.imshow(mask[0])
    plt.show()
    plt.imshow(mask[20])
    plt.show()
    plt.imshow(mask[40])
    plt.show()

path = os.getcwd() + "/data/processed/test/" 

def predict():
    net = unet_model_3d((1,64,64,64))
    net.load_weights("./data/logs/network_weights_loss.h5")
    
    for filename in os.listdir(path):

        f = h5py.File( path + filename , 'r')    
        amount_of_data = len(f['images/images'][:20])
        prediction_images = np.array(np.reshape(f['images/images'][:20], (amount_of_data,1,64,64,64)))
        prediction_labels = np.array(np.reshape(f['labels/labels'][:20], (amount_of_data,1,64,64,64)))
        predictions = (net.predict(prediction_images, batch_size=amount_of_data, verbose=1))   
        print(net.evaluate_generator(DataGenerator([0])))
        

        for i  in range(len(predictions)):
            dc = distance.dice(np.reshape(prediction_labels[i], (-1,)), np.reshape(predictions[i], (-1,)))
            if len(np.nonzero(prediction_labels[i])[1]) != 0:
                visualize_mask(predictions[i][0])
                visualize_mask(prediction_labels[i][0])
            print(dc)

    
    