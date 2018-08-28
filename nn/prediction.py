import os
import numpy as np
import tables
from .model import unet_model_3d
from .generator import DataGenerator, PredictGenerator 
from .metrics import dice_coefficient
path = os.getcwd() + "/data/processed/test/" 

def predict():

    net = unet_model_3d((1,64,64,64))
    net.load_weights("./data/logs/network_weights.h5")
    
    for filename in os.listdir(path):
        f = h5py.File( path + filename , 'r')
        prediction_data =  f['images/images']    


    predictions = net.predict(prediction_data, verbose=1)   
    
    for pred in predictions:
        dice_coefficient(truth,pred)

    
    