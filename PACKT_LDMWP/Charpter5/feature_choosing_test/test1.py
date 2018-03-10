import numpy as np
X=np.arange(0,30).reshape(10,3)#function arange() is the abbreviations of array range witch can be used as range() to set array numbers
# print(X)#we can consider the arry X as a dataset for 10 instances each of witch has 3 features
X[:,1]=-1#change the element in the second column into 1
# print(X)#the second feature for each instance is 1. After the adjustment the variance of feature1, 3 is much bigger than feature2


from sklearn.feature_selection import VarianceThreshold
vt=VarianceThreshold()#create an instance witch can select a feature whose variance is big enougth
Xt=vt.fit_transform(X)
# print(Xt)#the function fit_transform() abandoned feature2
# print(vt.variances_)#.variances_ can calculate the variance for each feature///([ 74.25   0.    74.25])
# Before we analyze the data we should obmit the feature whose variance is 0, otherwise the whole process will be slowdown





