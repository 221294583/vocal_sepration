import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.io import wavfile

def lstm(sample_length,features):
    model=Sequential()
    model.add(LSTM(2,input_shape=(sample_length,features),return_sequences=True,activation='relu'))
    model.add(Dense(features))
    model.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(learning_rate=1,beta_1=0.9,beta_2=0.999),metrics=['accuracy'])
    return model

def save_model(model,name):
    model.save(name)

def load_model(name):
    return keras.models.load_model(name)


