# -*- coding: utf-8 -*-

import numpy as np
import os
from src.models import AbstractModel
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, Input
from keras.layers import LSTM, Concatenate, Bidirectional
from keras.models import Model
import keras.backend as K
import keras

class PositionModel_embeded(AbstractModel):

    def __init__(self, config, len_seq_train, len_seq_pred, output_dim):

        self.save_directory = config['save_directory']
        self.model_name = config['model_name']
        self.len_seq_pred = len_seq_pred
        self.len_seq_train = len_seq_train
        self.output_dim = output_dim

        super().__init__(self.save_directory, self.model_name)
        
        self.config = config
        self.build_model()

    def euclidian_distance(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(K.clip(y_true*100, -1, 101) - K.clip(y_pred*100, -1, 101)), axis = -1))

    def build_model(self):

        COORD_INPUT_SHAPE = (self.len_seq_train, 4)
        EVENT_INPUT_SHAPE = (self.len_seq_train, )
        POSITION_INPUT_SHAPE = (self.len_seq_train, )
        ZONE_INPUT_SHAPE = (self.len_seq_train, )

        EVENT_EMBEDED_SIZE = self.config['EVENT_EMBEDED_SIZE']
        POSITION_EMBEDED_SIZE = self.config['POSITION_EMBEDED_SIZE']
        ZONE_EMBEDED_SIZE = self.config['ZONE_EMBEDED_SIZE']

        EVENT_EMBEDED_OUTPUT = self.config['EVENT_EMBEDED_OUTPUT']
        POSITION_EMBEDED_OUTPUT = self.config['POSITION_EMBEDED_OUTPUT']
        ZONE_EMBEDED_OUTPUT = self.config['ZONE_EMBEDED_OUTPUT']

        LSTM_SIZE_1 = self.config['LSTM_SIZE_1']
        LSTM_SIZE_2 = self.config['LSTM_SIZE_2']

        OUTPUT_DIM = self.output_dim

        coord_input = Input(shape = COORD_INPUT_SHAPE, name = "input_coord")
        position_input = Input(shape = POSITION_INPUT_SHAPE, name = "input_position")
        event_input = Input(shape = EVENT_INPUT_SHAPE, name = "input_event")
        zone_input = Input(shape = ZONE_INPUT_SHAPE, name = "input_zone")

        x_position = Embedding(input_dim = POSITION_EMBEDED_SIZE, output_dim = POSITION_EMBEDED_OUTPUT, name = "embeded_position")(position_input)
        print(x_position._keras_shape)

        x_event = Embedding(input_dim = EVENT_EMBEDED_SIZE, output_dim = EVENT_EMBEDED_OUTPUT, name = "embeded_event")(event_input)
        print(x_event._keras_shape)

        x_zone = Embedding(input_dim = ZONE_EMBEDED_SIZE, output_dim = ZONE_EMBEDED_OUTPUT, name = "embeded_zone")(zone_input)
        print(x_zone._keras_shape)

        x = Concatenate(axis = -1, name = "concatenate")([coord_input, x_position, x_event, x_zone])
        print(x._keras_shape)

        x = LSTM(LSTM_SIZE_1, return_sequences=True, name = "lstm_1")(x)
        print(x._keras_shape)

        x = LSTM(LSTM_SIZE_2, name = "lstm_2")(x)
        pred = Dense(OUTPUT_DIM)(x)

        model = Model([coord_input, position_input, event_input, zone_input], pred)
        optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer = optimizer, loss = 'mse', metrics = [self.euclidian_distance])
        
        self.model = model


class PositionModel_events(PositionModel_embeded):

    def build_model(self):

        COORD_INPUT_SHAPE = (self.len_seq_train, 4)

        LSTM_SIZE_1 = self.config['LSTM_SIZE_1']
        LSTM_SIZE_2 = self.config['LSTM_SIZE_2']

        OUTPUT_DIM = self.output_dim

        coord_input = Input(shape = COORD_INPUT_SHAPE)

        x = LSTM(LSTM_SIZE_1, return_sequences=True, name = "lstm_1")(coord_input)
        print(x._keras_shape)

        x = LSTM(LSTM_SIZE_2, name = "lstm_2")(x)
        x = Dropout(0.2)(x)

        pred = Dense(OUTPUT_DIM)(x)

        model = Model(coord_input, pred)
        optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(optimizer = optimizer, loss = 'mse', metrics = [self.euclidian_distance])
        
        self.model = model


class PositionModel_embeded_bid(AbstractModel):

    def __init__(self, config, len_seq_train, len_seq_pred, output_dim):

        self.save_directory = config['save_directory']
        self.model_name = config['model_name']
        self.len_seq_pred = len_seq_pred
        self.len_seq_train = len_seq_train
        self.output_dim = output_dim

        super().__init__(self.save_directory, self.model_name)
        
        self.config = config
        self.build_model()

    def euclidian_distance(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(K.clip(y_true*100, -1, 101) - K.clip(y_pred*100, -1, 101)), axis = -1))

    def build_model(self):

        COORD_INPUT_SHAPE = (self.len_seq_train, 4)
        EVENT_INPUT_SHAPE = (self.len_seq_train, )
        POSITION_INPUT_SHAPE = (self.len_seq_train, )
        ZONE_INPUT_SHAPE = (self.len_seq_train, )

        EVENT_EMBEDED_SIZE = self.config['EVENT_EMBEDED_SIZE']
        POSITION_EMBEDED_SIZE = self.config['POSITION_EMBEDED_SIZE']
        ZONE_EMBEDED_SIZE = self.config['ZONE_EMBEDED_SIZE']

        EVENT_EMBEDED_OUTPUT = self.config['EVENT_EMBEDED_OUTPUT']
        POSITION_EMBEDED_OUTPUT = self.config['POSITION_EMBEDED_OUTPUT']
        ZONE_EMBEDED_OUTPUT = self.config['ZONE_EMBEDED_OUTPUT']

        LSTM_SIZE_1 = self.config['LSTM_SIZE_1']
        LSTM_SIZE_2 = self.config['LSTM_SIZE_2']

        OUTPUT_DIM = self.output_dim

        coord_input = Input(shape = COORD_INPUT_SHAPE, name = "input_coord")
        position_input = Input(shape = POSITION_INPUT_SHAPE, name = "input_position")
        event_input = Input(shape = EVENT_INPUT_SHAPE, name = "input_event")
        zone_input = Input(shape = ZONE_INPUT_SHAPE, name = "input_zone")

        x_position = Embedding(input_dim = POSITION_EMBEDED_SIZE, output_dim = POSITION_EMBEDED_OUTPUT, name = "embeded_position", trainable = False)(position_input)
        print(x_position._keras_shape)

        x_event = Embedding(input_dim = EVENT_EMBEDED_SIZE, output_dim = EVENT_EMBEDED_OUTPUT, name = "embeded_event", trainable = False)(event_input)
        print(x_event._keras_shape)

        x_zone = Embedding(input_dim = ZONE_EMBEDED_SIZE, output_dim = ZONE_EMBEDED_OUTPUT, name = "embeded_zone", trainable = False)(zone_input)
        print(x_zone._keras_shape)

        x = Concatenate(axis = -1, name = "concatenate")([coord_input, x_position, x_event, x_zone])
        print(x._keras_shape)

        x = Bidirectional(LSTM(LSTM_SIZE_1, return_sequences=True, name = "bilstm_1"))(x)
        print(x._keras_shape)

        x = Bidirectional(LSTM(LSTM_SIZE_2, return_sequences=False, name = "bilstm_2"))(x)
        print(x._keras_shape)

        x = Dropout(0.2)(x)

        pred = Dense(OUTPUT_DIM)(x)
        print(pred._keras_shape)


        model = Model([coord_input, position_input, event_input, zone_input], pred)
        # optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(optimizer = optimizer, loss = 'mse', metrics = [self.euclidian_distance])
        
        self.model = model
