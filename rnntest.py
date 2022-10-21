import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.io import wavfile
import prepro
import build
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split

x_header='C:\\Users\\a\\PycharmProjects\\midi\\x\\flac\\'
y_header='C:\\Users\\a\\PycharmProjects\\midi\\y\\vocal\\'
temp=list(range(1,11))
temp=list(map(str,temp))
x_names=prepro.generate_nemalist(temp,x_header)
y_names=prepro.generate_nemalist(temp,y_header)
x_data,y_data,samplerate=prepro.get_data(x_names,y_names)
time=30
x_data=prepro.complement(x_data,time,samplerate)
y_data=prepro.complement(y_data,time,samplerate)
x_data,f,t=prepro.fourier(x_data,samplerate)
y_data,f,t=prepro.fourier(y_data,samplerate)
#x_data=prepro.diminish(x_data,0.0001)
#y_data=prepro.diminish(y_data,0.0001)

x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.1)
print(x_train,y_train,x_test,y_test)
sample_length=x_data[0].shape[0]
features=x_data[0].shape[1]
model=build.lstm(sample_length,features)
print(model.summary())
model.fit(x_train,y_train,epochs=2206,batch_size=3)

pre=model.predict(x_test)
#print(testX[0],testy[0])
prepro.to_show(f,t,x_test[0].T)
prepro.to_show(f,t,pre[0].T)
prepro.to_show(f,t,y_test[0].T)
