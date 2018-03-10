import numpy as np
import csv

##create a blank matrix
X=np.zeros((351,34),dtype='int')
y=np.zeros((351,),dtype='bool')#351 items in total
# print(X)
# print(Y)

##to input the data and make adjustment as well
data_filename='/home/konroy/PycharmProjects/PACKT_LDMWP/Charpter2/Ionosphere/ionosphere.data'
with open(data_filename,'r')as input_file:
    reader=csv.reader(input_file)
    for i ,row in enumerate(reader):
        # print("i: {0} ,row: {1}".format(i,row))
        data=[float(datum) for datum in row[:-1]]#from 0 to -1 (without -1)[-1 equals to the last one ('g' or 'b')]
        # print(data)
        X[i]=data
        y[i]=row[-1]=='g'#row[-1]'s value range from 'g' to 'b' .  The operator return boolean value

##then split the training set into training set and testing set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=14)

##then introduct a class KNeighbours and create an instance
from sklearn.neighbors import  KNeighborsClassifier
estimator=KNeighborsClassifier()

##after that use the modules to analyze the data
estimator.fit(X_train,y_train)

##lastly test the algorithmn(testing methor1)
y_predicted=estimator.predict(X_test)
accuracy=np.mean(y_test==y_predicted)*100
print("The accuracy(testing methor1) is {0:.1f}%".format(accuracy))

##lastly test the algorithmn(testing methor2)
from sklearn.cross_validation import cross_val_score
scores=cross_val_score(estimator,X,y,scoring='accuracy')
average_accuracy=np.mean(scores)*100
print("The accuracy(testing methor2) is {0:.1f}%".format(average_accuracy))

##end
