# iisc-mlsp

Observations:

1: loading the train image into numpy array

2: Dimension reduction using PCA. since D = 101*101 and N = 20. hence NxN S matrix for eigen vector X'u

3: 20 eigenvalues plotted. I saw that 19 values are more significant

4: dimension reduction from 101*101 to 19 [ 19 eigen vectors and 19 eigenvalues]

5: Now using LDA, datasets are projected into one dimesion vector

6: I found that for -ve value of LDA projection gives to happy face and +ve value of LDA projection gives to sad face

7: so threshold is taken as 0

8: for test data we got only 1 sad face result as happy. false positive is 1.


