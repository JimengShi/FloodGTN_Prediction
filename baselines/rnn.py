#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : WaLeF
@ FileName: rnn.py
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
from tensorflow.keras.layers import LSTM, SimpleRNN, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.models import load_model



def rnn(input_shape, rnn_unit1, rnn_unit2, rnn_unit3, l1_reg, l2_reg, dropout, masked_value):
    """
    l1_reg: 0
    l2_reg: 1e-5
    """
    inputs = keras.Input(shape=(input_shape))
    
    masked_inputs = layers.Masking(mask_value=masked_value)(inputs)
    
    x = SimpleRNN(rnn_unit1, 
                  activation='relu', 
                  kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
                  recurrent_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
                  return_sequences=True)(masked_inputs)
    x = Dropout(dropout)(x)
#     x = SimpleRNN(rnn_unit2, 
#                   activation='relu', 
#                   kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
#                   recurrent_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
#                   return_sequences=True)(x)
#     x = Dropout(dropout)(x)
#     x = SimpleRNN(rnn_unit3, 
#                   activation='relu', 
#                   kernel_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
#                   recurrent_regularizer=keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg),
#                   return_sequences=True)(x)
    x = layers.Flatten()(x)

    outputs = layers.Dense(96)(x)

    rnn_model = Model(inputs=inputs, outputs=outputs)
    #rnn_model.summary()


    return rnn_model
