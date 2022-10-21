import usual
import prepro
import uNetBuild
import numpy as np
from sklearn.model_selection import train_test_split
import os
from scipy.signal import istft
import database

con=['enter the X folder name','enter the X file name group','enter the number of the files',
     'enter the y folder name','enter the y file name group','enter the ratio of test samples',
     'enter epoch','enter batch size','Do you want to train(Y for training/N for skipping training)',
     'Do you want to predict(Y/N)','enter a filename','enter a wav filename',
     'EXIT?(Y/N)']

def main():
    path='C:\\Users\\a\\PycharmProjects\\midi\\checkpoint\\ramzy.h5'
    is_train=input('train?(Y/N)')
    if is_train=='y' or is_train=='Y':
        db = database.database()
        its = db.read_sub()
        print('there are {} subfolders'.format(its))
        db.random_pick()
        db.load_waveform()
        db.wave_to_complex_spec()
        db.complex_to_mag()
        db.time_fit()
        db.mk_train_unet()
        path = db.model_fit()

    ramzy=database.ramzy(path=input('name a file to predict'))
    ramzy.wave_to_complex_spec()
    ramzy.cutter()
    ramzy.complex_to_mag()
    ramzy.mk_predict_unet(autopath=path)
    ramzy.apply_mask()
    ramzy.inv()
    '''
    test_size=input(con[5])
    x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=float(test_size))
    print(x_train.shape,y_train.shape)

    judger,filename=usual.is_load()
    if judger:
        model=usual.load(filename,model)
    else:
        model=uNetBuild.unet()

    is_training=input(con[8])
    if is_training=='N' or is_training=='n':
        return 1

    print(np.mean(x_train),np.mean(y_train))
    epochs=int(input(con[6]))
    batch_size=int(input(con[7]))
    model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs)
    #model.evaluate(x_data,y_data,batch_size=batch_size)
    judger,filename=usual.is_save()

    if judger:
        usual.save(model,path=filename if filename!=0 else 'ramzy.h5')

    pre=model.predict(x_test)
    pre=np.squeeze(pre[0]).T
    to_chart=np.squeeze(y_test[0]).T
    print(pre)
    prepro.to_show(f,t,to_chart)
    prepro.to_show(f,t,pre)

    is_pre=input(con[9])
    if is_pre=='N' or is_pre=='n':
        return 0

    datafile=input(con[10])
    data=prepro.pre_load(datafile)
    data=prepro.pre_fourier(data)
    outcome=model.predict(data)
    outcome=np.squeeze(outcome)
    temp=outcome.shape[0]
    buffer=[outcome[i] for i in range(temp)]
    buffer=[i.T for i in buffer]
    buffer=np.hstack(buffer)
    n=2046
    buffer=istft(buffer,fs=44100,window='blackman',nperseg=n,noverlap=n*0.75,nfft=n)[1]
    print(type(buffer))
    wav_name=input(con[11])
    prepro.save_wave(''.join([wav_name,'.wav']),buffer)

    is_exit=input(con[12])
    if is_exit=='Y' or is_exit=='y':
        return 0

    return 1

'''
if __name__=='__main__':
    tik=1
    while tik:
        tik=main()
