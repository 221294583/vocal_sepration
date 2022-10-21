from keras.models import *
import prepro
import numpy as np
import os

con=['Do you want to save your model and its weights?(Y/N)','Do you want to customize the filename(Y/N)',
      'Please enter the filename','Do you want to load an existing model?(Y/N)','enter the file name of model']

def lan_version():
    print('pick your language')
    list=['english','chinese']
    print(' or '.join(list))
    temp=input()
    if temp=='chinese':
        return 0
    else:
        return 1

def save(model,path):
    model.save(path)

def load(path,model):
    model.load_weight(path,by_name=True)
    model.summary()
    return model

def load_data(x_pathname,x_namelist,y_pathname,y_namelist):
    x_namen=prepro.generate_nemalist(x_namelist,x_pathname)
    y_namen=prepro.generate_nemalist(y_namelist,y_pathname)
    x_data,y_data,samplerate=prepro.get_data(x_namen,y_namen)
    return x_data,y_data,samplerate

def is_save():
    print(con[0])
    temp=input()
    if temp=='Y' or temp=='y':
        buffer=input(con[1])
        name=0
        if buffer=='Y' or buffer=='y':
            name=input(con[2])
        return 1,name
    else:
        return 0,0

def is_load():
    temp=input(con[3])
    print(temp)
    if temp=='Y' or temp=='y':
        print(temp)
        name=input(con[4])
        return 1,name
    else:
        return 0,0

def get_namen(language):
    con =['enter the X folder name', 'enter the X file name group', 'enter the number of the files',
            'enter the y folder name', 'enter the y file name group', 'enter the ratio of test samples',
            'enter epoch', 'enter batch size']
    judger=input('default setting?(Y/N)')
    if judger=='Y' or judger=='y':
        return os.path.abspath('.\\x\\flac'),list(range(1,11)),os.path.abspath('.\\y\\vocal'),list(range(1,11))
    print(con[language][0])
    x_pathname=input()
    print(x_pathname)
    print(con[language][2])
    temp=int(input())
    print(temp)
    print(con[language][1])
    x_namelist=[]
    for i in range(temp):
        x_namelist.append(input())
    print(con[language][3])
    y_pathname = input()
    print(con[language][4])
    y_namelist = []
    for i in range(temp):
        y_namelist.append(input())
    return x_pathname,x_namelist,y_pathname,y_namelist
