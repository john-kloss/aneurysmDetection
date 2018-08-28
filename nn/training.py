import math
from keras import optimizers, losses
from .model import unet_model_3d
from keras import backend as K
import keras.callbacks 
import numpy as np
from .generator import DataGenerator 
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def train_model(model=None, model_file=None, data=None, steps_per_epoch=1, validations_steps=3, initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=500,
                learning_rate_patience=20, early_stopping_patience=None):
    
    
    
    net = unet_model_3d((1,64,64,64))
    net.load_weights("./data/logs/network_weights.h5")

    training_data = DataGenerator([1,2,3,4,5,6,7,8,9])
    validation_data = DataGenerator([10])
    

    # call = keras.callbacks.ModelCheckpoint("./data/logs/network_weights"+str(datetime.now())+".h5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)

    net.fit_generator(generator=training_data, steps_per_epoch=None, epochs=25, verbose=1, validation_data=validation_data, 
    validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=True, shuffle=True, 
    initial_epoch=0, callbacks=None)

    net.save_weights("./data/logs/network_weights.h5")

