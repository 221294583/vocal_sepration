import numpy as np
import os
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as backend


def unet(input_tensor,name_set,is_predict=False):
    #inputs = Input(input_size)
    #initial=tf.keras.initializers.he_normal
    conv1=Conv2D(16, 5, strides=(2,2),activation='relu', padding='same',name=name_set[0])(input_tensor)
    batch1 = BatchNormalization(axis=-1,name=name_set[1])(conv1)

    conv2=Conv2D(32, 5,strides=(2,2), activation='relu', padding='same',name=name_set[2])(batch1)
    batch2 = BatchNormalization(axis=-1,name=name_set[3])(conv2)

    conv3=Conv2D(64, 5,strides=(2,2), activation='relu', padding='same',name=name_set[4])(batch2)
    batch3 = BatchNormalization(axis=-1,name=name_set[5])(conv3)

    conv4=Conv2D(128, 5,strides=(2,2), activation='relu', padding='same',name=name_set[6])(batch3)
    batch4 = BatchNormalization(axis=-1,name=name_set[7])(conv4)

    conv5=Conv2D(256, 5,strides=(2,2), activation='relu', padding='same',name=name_set[8])(batch4)
    batch5 = BatchNormalization(axis=-1,name=name_set[9])(conv5)

    conv6=Conv2D(512, 5,strides=(2,2), activation='relu', padding='same',name=name_set[10])(batch5)
    #batch6 = BatchNormalization(axis=-1,name=name_set[11])(conv6)

    dec1=Conv2DTranspose(256,5,strides=[2,2],activation='relu',padding='same',name=name_set[12])(conv6)
    dec1=BatchNormalization(axis=-1,name=name_set[13])(dec1)
    drop1=Dropout(0.5,name=name_set[14])(dec1)
    merge1=Concatenate(axis=-1,name=name_set[15])([conv5,drop1])

    dec2=Conv2DTranspose(128,5,strides=[2,2],activation='relu',padding='same',name=name_set[16])(merge1)
    dec2=BatchNormalization(axis=-1,name=name_set[17])(dec2)
    drop2=Dropout(0.5,name=name_set[18])(dec2)
    merge2=Concatenate(axis=-1,name=name_set[19])([conv4,drop2])

    dec3=Conv2DTranspose(64,5,strides=[2,2],activation='relu',padding='same',name=name_set[20])(merge2)
    dec3=BatchNormalization(axis=-1,name=name_set[21])(dec3)
    drop3=Dropout(0.5,name=name_set[22])(dec3)
    merge3=Concatenate(axis=-1,name=name_set[23])([conv3,drop3])

    dec4=Conv2DTranspose(32,5,strides=[2,2],activation='relu',padding='same',name=name_set[24])(merge3)
    dec4=BatchNormalization(axis=-1,name=name_set[25])(dec4)
    merge4=Concatenate(axis=-1,name=name_set[26])([conv2,dec4])

    dec5=Conv2DTranspose(16,5,strides=[2,2],activation='relu',padding='same',name=name_set[27])(merge4)
    dec5=BatchNormalization(axis=-1,name=name_set[28])(dec5)
    merge5=Concatenate(axis=-1,name=name_set[29])([conv1,dec5])

    dec6=Conv2DTranspose(1,5,strides=[2,2],activation='relu',padding='same',name=name_set[30])(merge5)
    dec6=BatchNormalization(axis=-1,name=name_set[31])(dec6)

    cult=Conv2D(1,5,dilation_rate=(2,2),activation='relu',padding='same',name=name_set[32])(dec6)

    #model = Model(input=inputs, output=outputs)
    #model.compile(optimizer=Adam(lr=0.001), loss='mean_absolute_error', metrics=['accuracy'])


    #model.summary()

    #if (pretrained_weights):
    #    model.load_weights(pretrained_weights)

    #est=tf.keras.estimator.model_to_estimator(keras_model=model)
    if is_predict:
        return cult
    return Multiply(name=name_set[33])([cult,input_tensor])