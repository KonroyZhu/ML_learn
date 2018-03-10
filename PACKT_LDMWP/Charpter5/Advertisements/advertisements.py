import pandas as pd
import os
import numpy as np
from collections import defaultdict

data_folder="/home/konroy/Documents/konroy/learning data mining with python"
data_filename=os.path.join(data_folder,"ad.data")

#Issue1: pandas will consider the features at the front whose type is number as string
#Issue2: the missing data is showed by a question mark, we have to change it into nan(not a number)
def convert_number(serise):
    column=[]
    for x in serise:
        try:
            column.append(float(x))
        except ValueError:
            column.append(np.nan)
    return column
convertors=defaultdict()
convertors[1558]=lambda x:1 if x.strip()=="ad." else 0#1558 is the last column witch act as a modules

ads=pd.read_csv(data_filename,header=None,converters=convertors)
for i in range(3):
    ads[i]=convert_number(ads[i])
column=[]
for i in ads[3]:
    if i !="?":
        column.append(i)
    else:
        column.append(np.nan)
ads[3]=column

for i in ads[4]:
    if i=="?":
        print("True")

##Next : extract matrix x(containing the content) and array y(containing the modules )
X=ads.drop(1558,axis=1).values
y=ads[1558]
# print(x[:5])

##Then, choose the best feature from X
from sklearn.decomposition import PCA
pca=PCA(n_components=5)


# Xd=pca.fit_transform(X)#ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
# print(Xd[:5])
