import os
import random
#import soundfile as sf
import librosa
import numpy as np
from keras.models import *
from keras.optimizers import *
import keras
from functools import partial
import tensorflow as tf
import uNetBuild

def get_layer_names():
    layer_names=list(range(34))
    layer_names=list(map(str,layer_names))
    appendix=['a','b','c','d','e']
    golem=[]
    for i in range(len(appendix)):
        golem.append([''.join([j,appendix[i]]) for j in layer_names])
    return golem

class database:
    def __init__(self):
        '''
        '''
        self.dict=['mixture','vocals','accompaniment','drum','bass']
        path=input('name a working folder')
        self.path=''.join([path,'\\train\\'])

    def read_sub(self):
        '''
        self.namelist:paths to songs in root folder
        :return:
        '''
        dirs=os.listdir(self.path)
        namelist=[]
        for name in dirs:
            print(name)
            namelist.append(name)
        self.namelist=[''.join([self.path,i]) for i in namelist]
        return len(self.namelist)

    def random_pick(self):
        '''
        self.picklist:paths to songs randomly picked
        :return:
        '''
        picks=input('how many samples do you want to pick?'
                    '(Caution:the number of samples you pick cannot be bigger than the length of namelist!)')
        self.picklist=random.sample(self.namelist,int(picks))

    def load_waveform(self):
        '''
        load waveforms from those wav file picked
        :return:
        '''
        self.channels=int(input('how many channels do you want to load?'
                           '2 for vocal and accompaniment'))
        if self.channels==2:
            self.st=[0,1,2]
        elif self.channels==4:
            self.st=[0,1,2,3]
        elif self.channels==5:
            self.st=[0,1,2,3,4]
        for j in self.st:
            get_namen=[''.join([i,'\\',self.dict[j],'.wav']) for i in self.picklist]
            if j==0:
                self.d_mix=[librosa.load(i,sr=44100)[0] for i in get_namen]
            if j==1:
                self.d_voc=[librosa.load(i,sr=44100)[0] for i in get_namen]
            if j==2:
                self.d_acc=[librosa.load(i,sr=44100)[0] for i in get_namen]
            if j==3:
                self.d_drm=[librosa.load(i,sr=44100)[0] for i in get_namen]
            if j==4:
                self.d_bas=[librosa.load(i,sr=44100)[0] for i in get_namen]
        print(((self.d_mix)[0]).shape)
        self.samplerate=44100
        print('samplerate:',self.samplerate)

    def wave_to_complex_spec(self):
        '''
        change all the waveforms into spectrogram
        :return:
        '''
        self.s_mix=np.expand_dims(np.array([librosa.stft(i,n_fft=2048,hop_length=512,win_length=2048,window='hann')
                                            for i in self.d_mix])[:,:1024,:],axis=-1)
        self.s_voc=np.expand_dims(np.array([librosa.stft(i,n_fft=2048,hop_length=512,win_length=2048,window='hann')
                                            for i in self.d_voc])[:,:1024,:],axis=-1)
        self.s_acc=np.expand_dims(np.array([librosa.stft(i,n_fft=2048,hop_length=512,win_length=2048,window='hann')
                                            for i in self.d_acc])[:,:1024,:],axis=-1)
        try:
            self.d_drm
        except AttributeError:
            pass
        else:
            self.s_drm=np.expand_dims(np.array([librosa.stft(i,n_fft=2048,hop_length=512,win_length=2048,window='hann')
                                            for i in self.d_drm])[:,:1024,:],axis=-1)
        try:
            self.d_bas
        except AttributeError:
            pass
        else:
            self.s_bas=np.expand_dims(np.array([librosa.stft(i,n_fft=2048,hop_length=512,win_length=2048,window='hann')
                                            for i in self.d_bas])[:,:1024,:],axis=-1)

    def complex_to_mag(self):
        self.s_mix=np.abs(self.s_mix)
        self.s_voc=np.abs(self.s_voc)
        self.s_acc=np.abs(self.s_acc)
        try:self.s_drm
        except AttributeError:pass
        else:
            self.s_drm=np.abs(self.s_drm)
        try:self.s_bas
        except AttributeError:pass
        else:
            self.s_bas=np.abs(self.s_bas)
        print(np.mean(self.s_mix),np.mean(self.s_voc))

    def logged(self):
        '''
        apply log10 to spectrogram
        :return:
        '''
        self.s_mix=(np.log10(self.s_mix))*10
        self.s_mix[self.s_mix<0]=0
        self.s_voc=(np.log10(self.s_voc))*10
        self.s_voc[self.s_voc<0]=0
        self.s_acc=(np.log10(self.s_acc))*10
        self.s_acc[self.s_voc<0]=0
        try:self.s_drm
        except AttributeError:pass
        else:
            self.s_drm=(np.log10(self.s_drm))*10
            self.s_drm[self.s_drm<0]=0
        try:self.s_bas
        except AttributeError:pass
        else:
            self.s_bas=(np.log10(self.s_bas))*10
            self.s_bas[self.s_bas<0]=0

    def time_fit(self):
        sess=tf.Session()
        with sess.as_default():
            self.s_mix=tf.image.resize(self.s_mix,[1024,1024]).eval()
            self.s_voc=tf.image.resize(self.s_voc,[1024,1024]).eval()
            self.s_acc=tf.image.resize(self.s_acc,[1024,1024]).eval()
            try:self.s_drm
            except AttributeError:pass
            else:
                self.s_drm=tf.image.resize(self.s_drm,[1024,1024]).eval()
            try:self.s_bas
            except AttributeError:pass
            else:
                self.s_bas=tf.image.resize(self.s_bas,[1024,1024]).eval()

    def get_data(self):
        '''

        :return:x data and y data for training
        '''
        '''
        namen=['self.s_mix','self.s_voc','self.s_acc','self.s_drm','self.s_bas']
        print('pick y data you want', '\n',)
        for i in range(5):
            print('{0} for {1}'.format([str(i+1),namen[i]]))
        gets=list(input())
        gets=[int(i)-1 for i in gets]
        '''
        return self.s_mix,self.s_voc

    def mk_train_unet(self):
        branches=len(self.st)-1
        print(f'branches:{branches}')
        namen=get_layer_names()
        golem=[]
        input_tensor=Input((1024,1024,1))
        for i in range(branches):
            golem.append(uNetBuild.unet(input_tensor,namen[i]))
        model=Model(inputs=input_tensor,outputs=golem)
        model.compile(optimizer=Adam(lr=0.001),loss='mean_absolute_error',metrics=['accuracy'])
        model.summary()
        is_load=input('do you want to load weights?')
        if is_load=='y' or is_load=='Y':
            portal=input('name a file to load')
            model.load_weights(portal,by_name=True)
        self.model=model

    def model_fit(self,autopath='C:\\Users\\a\\PycharmProjects\\midi\\checkpoint\\ramzy.h5'):
        if len(self.st)==3:
            golem=[self.s_voc,self.s_acc]
        elif len(self.st)==4:
            golem=[self.s_voc,self.s_acc,self.s_drm]
        elif len(self.st)==5:
            golem=[self.s_voc,self.s_acc,self.s_drm,self.s_bas]
        epoch=int(input('assign epochs times'))
        batch=int(input('assign batch size'))
        gala=keras.callbacks.ModelCheckpoint(filepath=autopath,save_weights_only=True,
                                                monitor='val_loss',mode='min',save_best_only=True)
        self.model.fit(x=self.s_mix,y=golem,epochs=epoch,batch_size=batch,validation_split=0.1,callbacks=[gala])
        return autopath

class ramzy():
    def __init__(self,path):
        self.orgin=librosa.load(path,sr=44100)[0]

    def wave_to_complex_spec(self):
        self.spec=librosa.stft(self.orgin,n_fft=2048,hop_length=512,win_length=2048,window='hann')[:1024,]

    def cutter(self):
        segments=np.ceil(self.spec.shape[1]/1024)
        print(segments,self.spec.shape[1]/1024)
        golem=[]
        for i in range(int(segments)):
            temp=self.spec[:,i*1024:(i+1)*1024]
            if i==segments-1:
                complement=temp.shape[1]
                complement=np.zeros([1024,(1024-complement)])
                temp=np.hstack([temp,complement])
                print('complement')
            print(temp.shape)
            golem.append(temp)
        self.temp=np.array(golem)
        self.temp=np.expand_dims(self.temp,axis=-1)
        print('predict data got shape:',self.temp.shape)

    def complex_to_mag(self):
        self.mag=np.abs(self.temp)

    def mk_predict_unet(self,autopath='C:\\Users\\a\\PycharmProjects\\midi\\checkpoint\\ramzy.h5'):
        self.shape=Input([1024,1024,1])
        branches=2
        namen=get_layer_names()
        golem=[]
        for i in range(branches):
            golem.append(uNetBuild.unet(self.shape,namen[i],is_predict=True))
        model=Model(inputs=self.shape,outputs=golem)
        model.load_weights(autopath,by_name=True)
        self.pre=model.predict(self.mag)
        self.pre=np.squeeze(self.pre)
        self.name_dict=['vocal','accompaniment','drum','bas']
        self.pre_dict={}
        for i in range(branches):
            self.pre_dict[self.name_dict[i]]=self.pre[i]

    def apply_mask(self):
        self.masks=[]
        for i in range(2):
            operator=self.pre_dict[self.name_dict[i]]
            print(operator.shape)
            buffer=[]
            for j in range(operator.shape[0]):
                buffer.append(operator[j])
            buffer=np.hstack(buffer)
            print(buffer.shape)
            buffer=buffer[:,:self.spec.shape[1]]
            self.masks.append(buffer)
        self.re_spec=[np.multiply(i,self.spec) for i in self.masks]
        print(self.re_spec[0].shape)
        complement=np.zeros([self.spec.shape[1],])
        self.re_spec=[np.vstack([complement,i]) for i in self.re_spec]

    def inv(self):
        self.re_wave=[librosa.istft(i,hop_length=512,win_length=2048,window='hann') for i in self.re_spec]
        path=input('name a folder to save outcomes')
        path=''.join([path,'\\'])
        for i in range(len(self.re_wave)):
            librosa.output.write_wav(''.join([path,self.name_dict[i],'.wav']),self.re_wave[i],sr=44100)
