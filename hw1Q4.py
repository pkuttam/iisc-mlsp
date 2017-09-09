import os
import scipy.misc as sm
import numpy as np
path_image = 'emotion_classification/train/';
files = os.listdir(path_image)
L = np.empty(shape=(101*101, len(files)), dtype='float64') 
for i in range(0,len(files)):
    img = sm.imread(path_image+files[i])
    imgCol = np.array(img, dtype='float64').flatten()  # flatten the 2d image into 1d
    L[:, i] = imgCol[:]

meanImg = np.floor(np.sum(L, axis=1) / len(files))

for i in range(0,len(files)):
    L[:,i] = L[:,i]-meanImg;

covMat = np.floor((np.matrix(L) * np.matrix(L).transpose())/len(files))
#highDimMat = np.matrix(L).transpose * np.matrix(L);
eigVal, eigVect = np.linalg.eigh(covMat)  

