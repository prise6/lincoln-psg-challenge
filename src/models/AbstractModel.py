# -*- coding: utf-8 -*-

from keras.models import load_model
import numpy as np
import os

class AbstractModel:
    def __init__(self, save_directory, model_name):
        self.save_directory = save_directory
        self.model = None
        self.model_name = model_name

    def save(self):
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        self.model.save_weights('{}/final_{}.hdf5'.format(self.save_directory, self.model_name))
        
    def load(self, which = None):
        which = 'final_{}'.format(self.model_name) if which is None else which
        self.model.load_weights('{}/{}.hdf5'.format(self.save_directory, which))

    def predict(self, x, batch_size = None, verbose = 0, steps = None, callbacks = None):
        return self.model.predict(x, batch_size, verbose, steps)

    def predict_one(self, x, batch_size = 1, verbose = 0, steps = None):
        x = np.expand_dims(x, axis = 0)
        return self.predict(x, batch_size, verbose, steps)



