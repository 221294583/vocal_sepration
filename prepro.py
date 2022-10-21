import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import math

def pre_load(filename):
    data=np.array([np.mean(wavfile.read(filename)[1],axis=1)])
    return data

def pre_fourier(data,n=2046,window='blackman'):
    spec=spectrogram(data,44100,nperseg=n,noverlap=n/2,nfft=n,window=window)[2].T
    print(spec.shape)
    temp=spec.shape[0]
    temp=math.ceil(temp/512)
    buffer=[]
    for i in range(temp):
        ram=spec[1024*i:1024*(i+1),:]
        if ram.shape[0]!=1024:
            plus=np.zeros((1024-ram.shape[0],ram.shape[1],ram.shape[2]))
            ram=np.vstack((ram,plus))
        print(ram.shape)
        buffer.append(ram)
    buffer=np.array(buffer)
    return buffer

def generate_nemalist(namen,loc):
    tail='.wav'
    namen=[''.join([loc,'\\',str(i),tail]) for i in namen]
    return namen

def get_data(xnamen,ynamen):
    x_data=np.array([np.mean(wavfile.read(i)[1],axis=1) for i in xnamen])
    y_data=np.array([np.mean(wavfile.read(i)[1],axis=1) for i in ynamen])
    samplerate=44100
    return x_data,y_data,samplerate

def complement(dataset,samplerate,time):
    cutter=time*samplerate
    for i in range(len(dataset)):
        if len(dataset[i]) >= cutter:
            dataset[i] = dataset[i][:cutter]
        elif len(dataset[i]) < cutter:
            buffer = np.zeros((cutter - len(dataset[i]),))
            dataset[i] = np.hstack((dataset[i], buffer))
    dataset=np.vstack(dataset)
    return dataset

def diminish(dataset,ratio):
    for i in range(len(dataset)):
        operator=(np.amax(dataset[i]))*ratio
        dataset[i]=dataset[i]
        dataset[i]=np.where(dataset[i]>=operator,dataset[i],0)
    return dataset

def fourier(dataset,samplerate,window='blackman',n=2046):
    f,t,trash=spectrogram(dataset[0],samplerate,nperseg=n,noverlap=n/2,nfft=n,window=window)
    buffer=[]
    for i in range(len(dataset)):
        temp=spectrogram(dataset[i],samplerate,nperseg=n,noverlap=n/2,nfft=n,window=window)[2].T
        temp=temp[:1024,:]
        temp=np.expand_dims(temp,axis=0)
        buffer.append(temp)
    buffer=np.vstack(buffer)
    print(buffer.shape)
    return buffer,f,t

def fourier_for_predict(data,samplerate,window='blackman',n=4095):
    f,t,z=spectrogram(data,samplerate,nperseg=n,noverlap=n/2,nfft=n,window=window)
    buffer=[]
    blocks=math.ceil(len(t)/512)
    for i in range(blocks):
        temp=z[512*i:512*(i+1),:]
        temp=np.expand_dims(temp,axis=0)
        buffer.append(temp)
    buffer=np.vstack(buffer)
    return buffer,f,t

def to_show(f,t,data):
    fig=plt.figure()
    plt.pcolormesh(f,t,data,shading='gouraud')
    plt.xlabel('discrete frequency')
    plt.ylabel('time')
    plt.show()

def save_wave(name,data):
    wavfile.write(name,44100,data)