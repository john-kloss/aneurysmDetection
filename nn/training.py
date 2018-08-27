import math
from keras import optimizers, losses
from .model import unet_model_3d
from keras import backend as K
import keras.callbacks 
import numpy as np
from keras.callbacks import LambdaCallback
from .generator import data_gen 
import tensorflow as tf
import os
from keras.callbacks import TensorBoard
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



def train_model(model=None, model_file=None, data=None, steps_per_epoch=1, validations_steps=3, initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=500,
                learning_rate_patience=20, early_stopping_patience=None):
    
    
    
    net = unet_model_3d((1,64,64,64))
    #net.load_weights("./data/tmp/logs/network_weights.h5")
    training_data = data_gen([0,1,2,3],1,40)
    validation_data = data_gen([4],1,10)
    #history = LossHistory()
    call= keras.callbacks.ModelCheckpoint("./data/tmp/logs/network_weights.h5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)

    net.fit_generator(generator=training_data, steps_per_epoch=10*4, epochs=10, verbose=1, validation_data=validation_data, 
    validation_steps=10, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, 
    initial_epoch=0, callbacks=[call])
    
    net.save_weights("./data/tmp/logs/network_weights.h5")
"""
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
"""

