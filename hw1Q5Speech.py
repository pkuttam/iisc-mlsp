import glob
import os
import numpy as np
import matplotlib
from scipy.fftpack import fft
from scipy.io import wavfile 

import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram

import librosa
import math

f1 = 'speech/clean.wav'
f2 = 'speech/noise.wav'
f3 = 'speech/noisy.wav'

# 1 sec 16000 samples
# 25 ms 16000/40 = 400 samples
# 10 ms 16000/100 = 160 shift
w = 400; #window
sh = 160; # shift
magScalePoint = 256;


#sr is sample rate
#sr1, XnoisyInt = wavfile.read('speech/noisy.wav')
#sr2, XcleanInt = wavfile.read('speech/clean.wav')
#sr3, XnoiseInt = wavfile.read('speech/noise.wav')
#machineInt = 2**
Xnoisy, sr1 = librosa.load('speech/noisy.wav',16000) # loading the wav file with sample frequency 16000
Xclean, sr2 = librosa.load('speech/clean.wav',16000)
Xnoise, sr3 = librosa.load('speech/noise.wav',16000)

shiftPointNoisy = range(0,len(Xnoisy),160) # sample stamp point like 0 160 320 ......
shiftPointClean = range(0,len(Xclean),160)
shiftPointNoise = range(0,len(Xnoise),160)

totFrameNoisy = len(shiftPointNoisy); #number of total sample windows (total number of stft)
totFrameClean = len(shiftPointClean);
totFrameNoise = len(shiftPointNoise);

spectroNoisy = np.zeros(shape=(totFrameNoisy,magScalePoint), dtype='float64') # memory allocation of stft
spectroClean = np.zeros(shape=(totFrameClean,magScalePoint), dtype='float64')
spectroNoise = np.zeros(shape=(totFrameNoise,magScalePoint), dtype='float64')  
halfCosine = np.hanning(w) # half cosine which will be multiplied in signal
count = 0

for i in shiftPointNoisy:
    if len(Xnoisy[i:])>=w:
        xfft = np.abs(fft(Xnoisy[i:i+w]*halfCosine,(magScalePoint)*2)) # symmetry of fft with bins=256
        spectroNoisy[count,:] = abs(xfft[0:256])
    else:
        sig = np.zeros(shape = w,dtype='float64')
        sig[0:len(Xnoisy[i:])]=Xnoisy[i:]
        xfft =  np.abs(fft(sig*halfCosine,(magScalePoint)*2))
        spectroNoisy[count,:] = abs(xfft[0:256])
    count = count+1

count = 0
for i in shiftPointClean:
    if len(Xnoisy[i:])>=w:
        xfft = np.abs(fft(Xclean[i:i+w]*halfCosine,(magScalePoint)*2)) # symmetry of fft with bins=256
        spectroClean[count,:] = abs(xfft[0:256])
    else:
        sig = np.zeros(shape = w,dtype='float64')
        sig[0:len(Xclean[i:])]=Xclean[i:]
        xfft =  np.abs(fft(sig*halfCosine,(magScalePoint)*2))
        spectroClean[count,:] = abs(xfft[0:256])
    count = count+1

#count=0
#for i in shiftPointNoise:
#    if len(Xnoisy[i:])>=w:
#        xfft = np.abs(fft(Xnoise[i:i+w]*halfCosine,(magScalePoint)*2)) # symmetry of fft with bins=256
#        spectroNoise[count,:] = abs(xfft[0:256])
#    else:
#        sig = np.zeros(shape = w,dtype='float64')
#        sig[0:len(Xnoise[i:])]=Xnoise[i:]
#        xfft =  np.abs(fft(sig*halfCosine,(magScalePoint)*2))
#        spectroNoise[count,:] = abs(xfft[0:256])
#    count = count+1

avNoiseInDB = np.log10(np.sum(np.divide(spectroNoisy,spectroClean),1)/256) 

## divide clean speech into 2 parts
trainCleanSpectro = spectroClean[0:totFrameClean/2,:]
testCleanSpectro = spectroClean[(totFrameClean/2):,:]

## mean of train data of clean speech
meanTrainCleanSpectro = np.matrix(np.sum(trainCleanSpectro,0)/len(trainCleanSpectro))

## variance of data
S = (trainCleanSpectro - meanTrainCleanSpectro).transpose()*(trainCleanSpectro - meanTrainCleanSpectro)

## eigenvalue of symmetry matrix
eigVal, eigVect = np.linalg.eigh(S)

idx = eigVal.argsort()[::-1]   
eigVal = eigVal[idx]
eigVect = eigVect[:,idx]

## dimension = 5
eigVal5D = eigVal[0:5]
eigVect5D = eigVect[:,0:5]

## projection
projSpectroClean = (spectroClean-meanTrainCleanSpectro)*eigVect5D
projSpectroNoisy = (spectroNoisy-meanTrainCleanSpectro)*eigVect5D

## reconstruction
reconsSpectroClean = projSpectroClean*eigVect5D.transpose() + meanTrainCleanSpectro
reconsSpectroNoisy = projSpectroNoisy*eigVect5D.transpose() + meanTrainCleanSpectro

##

avNoiseReconsInDB = np.log10(np.sum(np.divide(reconsSpectroNoisy,reconsSpectroClean),1)/256) 

plt.plot(avNoiseInDB)
plt.hold;
plt.plot(avNoiseReconsInDB,'r')
plt.show()
