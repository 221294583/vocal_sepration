import numpy as np
import uNetBuild
import tensorflow as tf
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
x_data=np.expand_dims(x_data,axis=-1)
y_data=np.expand_dims(y_data,axis=-1)

x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2)
sample_length=x_data[0].shape[0]

model=uNetBuild.unet()
model.fit(x_train,y_train,batch_size=2,epochs=60)

pre=model.predict(x_test)
print(pre[0].shape)
prepro.to_show(f,t,(x_test[0].T).squeeze)
prepro.to_show(f,t,(pre[0].T).squeeze)
prepro.to_show(f,t,(y_test[0].T).squeeze)