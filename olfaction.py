import numpy as np
import midiutil.MidiFile as mmi
import copy
import struct
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class mues:

    def __init__(self,Sxx,f):
        #客制数据
        self.dataset=Sxx.T
        self.freqstream=f
        temp = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        notes=[]
        for i in range(9):
            buffer=[''.join([b,str(i)]) for b in temp]
            notes+=buffer
        notes=np.array(notes)
        freq=np.loadtxt('note.txt')
        freq=freq.T
        freq=freq.ravel()
        hex_code=np.loadtxt('hexa_note.txt')
        hex_code=hex_code.T
        hex_code=hex_code.ravel()
        ram=np.vstack((freq,hex_code))
        ram=np.vstack((ram,notes))
        #频率信息;0代表频率;1代表midi十六进制编码;2代表音符
        self.judger=ram

    def beats(self):
        '''
        查找节拍信息
        :return:
        '''
        operator=self.dataset
        sum=np.sum(operator,axis=1)
        placebo=np.zeros(len(sum))
        for i in range(len(sum)-1):
            placebo[i+1]=sum[i]
        delta=sum-placebo
        beats=[]
        for i in range(len(delta)):
            if delta[i]>0 and delta[i+1]<0:
                beats.append(i)
        beats=np.array(beats)
        self.beat_time_seq=beats   #节拍所在点的时序数组

    def find_closest_note(self,freq):
        '''
        查找和一个给定频率最接近的频率的在self.judger中的序号
        :param freq: 单个频率值
        :return: 最接近的频率值在self.judger中对应的序号
        '''
        operator=self.judger[0]
        operator=operator-freq
        operator=abs(operator)
        sequence=np.argmin(operator)
        return sequence

    def match(self):
        for i in self.beat_time_seq:
            temp=self.dataset[i]
            temp_1=self.dataset[i+1]
            buffer=np.argsort(temp)[::-1]
            buffer=buffer[0:3]
            buffer_1=np.argsort(temp_1)[::-1]
            buffer_1=buffer_1[0:3]
            operator=np.array([temp[i] for i in buffer])   #峰值前面最高的三个强度
            operator_freq=np.array([self.freqstream[i] for i in buffer])   #峰值前面最高的三个强度对应的频率
            operator_1=np.array([temp_1[i] for i in buffer_1])   #峰值后面最高的三个强度
            operator_freq_1=np.array([self.freqstream[i] for i in buffer_1])   #峰值后面最高的三个强度对应的频率
            