#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : WaLeF
@ FileName: cnn.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 6/18/22 08:57
"""

from pandas import DataFrame
from pandas import concat
from pandas import concat, read_csv
from tensorflow import keras
from tensorflow.keras import Model, Input, layers
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv1D, SimpleRNN, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.models import load_model



def cnn(input_shape, cnn_unit1, cnn_unit2, cnn_unit3, kernel_size, dropout, pool_size, masked_value):
    inputs = keras.Input(shape=(input_shape))
    
    masked_inputs = layers.Masking(mask_value=masked_value)(inputs)
    
    x = Conv1D(cnn_unit1, 
               kernel_size=kernel_size, 
               activation='relu', 
               padding="same")(masked_inputs)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Dropout(dropout)(x)
    x = Conv1D(cnn_unit2, 
               kernel_size=kernel_size, 
               activation='relu', 
               padding="same")(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Conv1D(cnn_unit3, 
               kernel_size=kernel_size, 
               activation='relu', 
               padding="same")(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Flatten()(x)

    outputs = Dense(96)(x)

    cnn_model = Model(inputs=inputs, outputs=outputs)
    #cnn_model.summary()


    return cnn_model
