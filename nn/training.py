import math
from keras import optimizers, losses

from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model


def train_model(model, model_file, data, steps_per_epoch, validation_steps,
                initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=500,
                learning_rate_patience=20, early_stopping_patience=None):
    
    optimizer = optimizers.Adam(lr=initial_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=learning_rate_drop,
                amsgrad=False)
    loss = losses.binary_crossentropy(y_true, y_pred)


    model.compile(optimizer, loss=loss, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=Non)
    
    model.fit(x=None, y=None, batch_size=1, epochs=10, verbose=1, callbacks=None, validation_split=0.1, 
                validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
                steps_per_epoch=None, validation_steps=None)
