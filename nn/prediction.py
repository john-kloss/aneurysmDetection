import os
import numpy as np
import tables
from .model import unet_model_3d
from .generator import DataGenerator, PredictGenerator 
from .metrics import dice_coefficient

def predict():
    
    # initialize test data generator without labels for predicting 
    prediction_data = 
    net = unet_model_3d((1,64,64,64))
    net.load_weights("./data/logs/network_weights.h5")

    predictions = net.predict(prediction_data, verbose=1)   
    
    for pred in predictions:
        dice_coefficient(truth,pred)

    