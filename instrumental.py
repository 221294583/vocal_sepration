import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from scipy.io import wavfile
import prepro

filename=input('请输入文件名:')
samplerate,dataset=wavfile.read(filename)

dataset=np.mean(dataset,axis=1)
print('持续时间:',len(dataset)/samplerate)

dataset=prepro.complement(dataset,samplerate)
print(dataset.shape)

