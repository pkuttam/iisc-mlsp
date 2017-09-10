import os
import scipy.misc as sm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

path_image = 'emotion_classification/train/';
files = os.listdir(path_image)
LD = np.empty(shape=(101*101, len(files)), dtype='float64') 
for i in range(0,len(files)):
    img = sm.imread(path_image+files[i])
    imgCol = np.array(img, dtype='float64').flatten()  # flatten the 2d image into 1d
    LD[:, i] = imgCol[:]

meanImg = (np.sum(LD, axis=1) / len(files))
L = np.empty(shape=(101*101, len(files)), dtype='float64') 
for i in range(0,len(files)):
    L[:,i] = LD[:,i]-meanImg;

#covMat = np.floor((np.matrix(L) * np.matrix(L).transpose())/len(files))
highDimMat = (np.matrix(L).transpose() * np.matrix(L))/len(files);
eigVal, eigVect = np.linalg.eigh(highDimMat)  

idx = eigVal.argsort()[::-1]   
eigVal = eigVal[idx] # lamda 
eigVect = eigVect[:,idx] # v


eigVectInDataDim = np.divide( (L*eigVect),np.sqrt(np.abs(len(files)*eigVal)));

plt.plot(eigVal)
plt.show()

# looking eigenvalues take K = 19 since all the other 101*101 -20 eigen values will be zeros
eigValDimRed = eigVal[0:19]
eigVectDimRed = eigVectInDataDim[:,0:19]

# project data

projectedData = np.matrix(LD).transpose() *eigVectDimRed # projected data is row wise

# lets label the data
# C1 is sad and C2 is happy . now looking the file order in files
#>>> files
#['subject02.sad.gif', 'subject13.sad.gif', 'subject04.sad.gif', 'subject12.happy.gif', 'subject06.happy.gif', 'subject05.sad.gif', 'subject13.happy.gif', 'subject03.sad.gif', 'subject07.sad.gif', 'subject11.sad.gif', 'subject01.happy.gif', 'subject04.happy.gif', 'subject10.sad.gif', 'subject12.sad.gif', 'subject06.sad.gif', 'subject09.sad.gif', 'subject10.happy.gif', 'subject07.happy.gif', 'subject02.happy.gif', 'subject09.happy.gif']

C1 = [0,1,2,5,7,8,9,12,13,14,15]
C2 = [3,4,6,10,11,16,17,18,19] 

XsadD = projectedData[C1,:]
XhappyD = projectedData[C2,:]

meanXsadD = np.sum(XsadD,0)/len(C1);
meanXhappyD = np.sum(XhappyD,0)/len(C2);

Xsad = XsadD - np.ones(shape=(len(C1),1),dtype='float64')*meanXsadD
Xhappy = XhappyD - np.ones(shape=(len(C2),1),dtype='float64')*meanXhappyD

SW = Xsad.transpose()*Xsad + Xhappy.transpose()*Xhappy

meanDiff = meanXhappyD-meanXsadD

w = np.linalg.inv(SW)*meanDiff.transpose()

wNormalized = w/np.linalg.norm(w)

XsadDProj = XsadD*wNormalized
XhappyDProj = XhappyD*wNormalized

############################# test data ###############################################
test_path_image = 'emotion_classification/test/';
filesTest = os.listdir(test_path_image)
LTestD = np.empty(shape=(101*101, len(filesTest)), dtype='float64') 

for i in range(0,len(filesTest)):
    img = sm.imread(path_image+files[i])
    imgCol = np.array(img, dtype='float64').flatten()  # flatten the 2d image into 1d
    LTestD[:, i] = imgCol[:]

#LSubMeanTest = np.empty(shape=(101*101, len(filesTest)), dtype='float64') 
#for i in range(0,len(filesTest)):
#    LSubMeanTest[:,i] = LTestD[:,i]-meanImg;
projectedTestData = np.matrix(LTestD).transpose() *eigVectDimRed
xtestProj = projectedTestData*wNormalized

threshold = 0
result = xtestProj<=threshold
#print(filesTest)
print("\nTrue indicates happy but False indicates sad\n")
#print(result)

for i in range(0,len(filesTest)):
    if(result[i]==True):
        print(filesTest[i] + "------> Happy")
    else:
        print(filesTest[i] + "------> Sad")
   

