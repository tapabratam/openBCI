# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:18:11 2017

@author: Rishav
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


SAMPLING_FREQUENCY=200
NUMBER_OF_SAMPLES=60
LOWER_CUT=10
HIGHER_CUT=40
NOTCH=50
ORDER=4
TAKE_FOI=True
NUMBER_OF_CHANNELS=8
TESTING=True
MODE='TRAINING'
#...................DATA.....................................


#....................Data acquisition Left................................
def data_left(x='C://Users//Rishav//Desktop//auto_left.csv',RETURN_Y=False):
    

    df_left_original=pd.read_csv(x)
    df_left=df_left_original[:NUMBER_OF_SAMPLES*SAMPLING_FREQUENCY]
    channels=[df_left['1'],df_left['2'],df_left['3'],df_left['4'],df_left['5'],df_left['6'],df_left['7'],df_left['8']]
    i=['1','2','3','4','5','6','7','8']
    y_left=df_left['label']
    y_left=y_left[:NUMBER_OF_SAMPLES]
    
    arr_left= []
    for ch,x in zip(channels,i):    
        ch=np.asarray(df_left[x],dtype=np.float64)
        ch.flatten()
        ch=ch.reshape(NUMBER_OF_SAMPLES,SAMPLING_FREQUENCY)
        arr_left.append(ch)
    arr_left=np.array(arr_left)
    if RETURN_Y:
        return y_left
    else:
        return arr_left
    
#.........................data acquisition Right.............................

def data_right(x='C://Users//Rishav//Desktop//auto_right.csv',RETURN_Y=False):
    df_right_original=pd.read_csv(x)
    df_right=df_right_original[:NUMBER_OF_SAMPLES*SAMPLING_FREQUENCY]
    channels=[df_right['1'],df_right['2'],df_right['3'],df_right['4'],df_right['5'],df_right['6'],df_right['7'],df_right['8']]
    i=['1','2','3','4','5','6','7','8']
    y_right=df_right['label']
    y_right=y_right[:NUMBER_OF_SAMPLES]
    arr_right= []
    for ch,x in zip(channels,i):    
        ch=np.asarray(df_right[x],dtype=np.float64)
        ch.flatten()
        ch=ch.reshape(NUMBER_OF_SAMPLES,SAMPLING_FREQUENCY)
        arr_right.append(ch)
    arr_right=np.array(arr_right)
    
    if RETURN_Y:
        return y_right
    else:
        return arr_right
#.......................data acquired...................................  



#........................BAND PASS FILTER..............................

def filtering(sequence,lower_cut=LOWER_CUT,higher_cut=HIGHER_CUT,order=ORDER,notch=NOTCH,return_both=False,fs=SAMPLING_FREQUENCY):
    
    nyq=0.5*fs #normalizing
    low=lower_cut/nyq
    high=higher_cut/nyq
    sequence=sequence/nyq
    Q=30 #Q-factor
    w0=notch/nyq #Normalizing the notch
    
    b_notch,a_notch=signal.iirnotch(w0,Q)         #Designing the NOTCH filter at 50Hz
    
    sequence_after_notch=signal.lfilter(b_notch,a_notch,sequence) #Filtering The 50Hz component
    
    b_bandpass,a_bandpass=signal.butter(order,[low,high],btype='band')
    
    sequence_after_bandpass=signal.lfilter(b_bandpass,a_bandpass,sequence)
    
    if return_both:
        return  sequence_after_notch,sequence_after_bandpass
    else:
        #print(sequence_after_bandpass.shape)
        return sequence_after_bandpass
    

#........................................................................


#..................FREQUENCY DOMAIN TRANSFORM.............................
def FFT(sequence,mag_2D=False,plot_data=False):
    
    arr=np.array(sequence)
    fft_coeff=np.fft.rfft(arr)
    real_part=np.real(fft_coeff)
    imag_part=np.imag(fft_coeff)
    mag=np.sqrt((real_part*real_part)+(imag_part*imag_part))
    
    if not TESTING:
        mag[:,:,0]=0#Removing THE DC COMPONENT
    elif mag_2D:
        mag[:,0]=0
    else:
        mag[0]=0
    print("FOURIER_TRANSFORM-'{mag}'".format(mag=mag.shape))
    if plot_data:
        plt.plot(mag)
    return mag
        
#...........................................................................


#......................CREATING THE DATA SET (BEFORE PCA).....................
def DEPRECIATED_ARR(TAKE_FOI=True):
    arr_left=data_left()
    arr_right=data_right()
    filtered_array_left=filtering(arr_left)
    frequency_array_left=FFT(filtered_array_left)
    
    filtered_array_right=filtering(arr_right)
    frequency_array_right=FFT(filtered_array_right)
    
    if TAKE_FOI:
        frequency_array_left=frequency_array_left[:,:,LOWER_CUT:HIGHER_CUT+1]
        frequency_array_right=frequency_array_right[:,:,LOWER_CUT:HIGHER_CUT+1]
        print("Final Frequency shape'{freq}'".format(freq=frequency_array_right.shape))
    return frequency_array_left,frequency_array_right 
#............................................................................



#.............................DATA AFTER PCA..............................


#.........................................................................


#.............................MODEL WITHOUT PCA..........................

#........................CREATING THE DATASET........................
def CREATE_DATASET(ns=NUMBER_OF_SAMPLES,nc=NUMBER_OF_CHANNELS):
    X_left=[]
    X_right=[]
    frequency_array_left,frequency_array_right=DEPRECIATED_ARR()
    for i in range(ns):
        x_data_left=[]
        for j in range(nc):
            x_data_left.append(frequency_array_left[j][i])
        x_data_left=np.array(x_data_left).flatten()
        X_left.append(x_data_left)
    X_left=np.array(X_left)
    
    print("SHAPE OF LEFT DATASET'{x}'".format(x=X_left.shape))
    
    for i in range(ns):
        x_data_right=[]
        for j in range(nc):
            x_data_right.append(frequency_array_right[j][i])
        x_data_right=np.array(x_data_right).flatten()
        X_right.append(x_data_right)
    X_right=np.array(X_right)
    
    print("SHAPE OF RIGHT DATASET'{x}'".format(x=X_right.shape))


    X=np.concatenate((X_left,X_right),axis=0)
    
    y_left=data_left(RETURN_Y=True)
    y_right=data_right(RETURN_Y=True)
    y_left=np.array(y_left)
    y_right=np.array(y_right)
    y=np.concatenate((y_left,y_right),axis=0)
    print("SHAPE OF FINAL DATASET X='{x}',Y='{Y}'".format(x=X.shape,Y=y.shape))
    
    
    return X,y



def PREPROCESS(sequence):
    data=np.array(sequence)
    if TESTING:
        data=data[:,:-1]
        data=data.transpose()
        filtered_data=filtering(data)
        frequency_data=FFT(filtered_data)
        if TAKE_FOI:
            frequency_data=frequency_data[:,LOWER_CUT:HIGHER_CUT+1]
    frequency_data=frequency_data.flatten()
    frequency_data=frequency_data.reshape(1,-1)
    return frequency_data 
