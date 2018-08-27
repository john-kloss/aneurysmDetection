import os
import numpy as np
import tables
from .model import unet_model_3d
from .generator import DataGenerator 


def predict():
    # generate the test data 
    test_data = DataGenerator([0])

    net = unet_model_3d((1,64,64,64))
    net.load_weights("./data/logs/network_weights.h5")
    pre = net.predict_generator(test_data, verbose=1)   
    print(pre)
    return net.evaluate_generator(test_data, verbose=1)