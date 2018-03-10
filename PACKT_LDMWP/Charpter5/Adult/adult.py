import pandas as pd
import os

data_folder="/home/konroy/Documents/konroy/learning data mining with python/Adult"
adult_filename=os.path.join(data_folder,"adult.data")
adult=pd.read_csv(adult_filename,header=None,names=["Age","Work-Class","fnlwgt","Education","Education-Num","Marital-Status",
                                                    "Occupation","Relationship","Race","Sex","Capital-Gain","Capital-Loss",
                                                    "Hours-Per-Week","Native-Country","Earning-Raw"])
adult.dropna(how='all',inplace=True)#delete the column that contain invalid element
# print(adult[:5])
# print(adult.columns)
# print(adult["Hours-Per-Week"].describe())#function describe() can show 'mean,standar,min,max'
# print(adult["Education-Num"].median())#function median() can find out the average
# print(adult["Work-Class"].unique())#function unique() can find out all the work class without duplication


X=adult[["Age","Education-Num","Capital-Gain","Capital-Loss","Hours-Per-Week"]].values
# print(X)
y=(adult["Earning-Raw"]=='>50').values
# print(y)


##methor1
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
transformer=SelectKBest(score_func=chi2,k=3)
Xt_chi2=transformer.fit_transform(X,y)
# print(Xt_chi2)
# print(transformer.scores_)#not a number ?


