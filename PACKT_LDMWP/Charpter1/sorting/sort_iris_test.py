from sklearn.datasets import  load_iris
dataset=load_iris()
X=dataset.data
y=dataset.target#represent the classification


import numpy as np
attribute_means=X.mean(axis=0)#to calculate the mean value(average) for each feature
X_d=np.array(X>=attribute_means,dtype='int')#to discretize the elements
print(X_d)

from collections import defaultdict
from operator import itemgetter

def train_feature_value(X_d,y_true,feature_index,value):
    # print("current_value (inside founction): {0}".format(value))
    class_counts=defaultdict(int)
    for sample ,y in zip(X_d,y_true):
        # print("y: "+str(y))#clssification label
        # print("sample: {0}".format(sample))#each instance
        if sample[feature_index]==value:
            class_counts[y]+=1
    sorted_class_counts=sorted(class_counts.items(),key=itemgetter(1),reverse=True)
    most_frequent_class=sorted_class_counts[0][0]#after sorting the first one is the largest one
    incorrect_predictions=[class_count for class_value,class_count in class_counts.items()
                           if class_value !=most_frequent_class ]
    error=sum(incorrect_predictions)#to collect how many predictions are incorrect
    return most_frequent_class,error

def train_on_feature(X_d,y_true,feature_index):
    values=set(X_d[:,feature_index])#{0, 1} value range for feature
    predictors={}
    errors=[]
    # print("values: {0}".format(values))

    for current_value in values:
        # print("current_value: {0}".format(current_value))
        most_frequent_class,error=train_feature_value(X_d,y_true,feature_index,current_value)
        # print("current_value: {0}".format(current_value))
        predictors[current_value]=most_frequent_class
        errors.append(error)
        total_error=sum(errors)
        return predictors,total_error#when the 'current_value' is  current_value which type own the less error
        #after adding return the current_value cannot be completely printed out but the founction did go throught it

def predict(X_test,model):
    variable=model['feature']
    predictor=model['predictor']
    ###
    print("Hello")
    print(Xd_test)
    for sample in X_test:
        print(sample)
    print([predictor[int(sample[variable])]for sample in X_test])
    ###
    y_predicted=np.array(int([predictor[int(sample[variable])]for sample in X_test]))
    # return y_predicted

from sklearn.cross_validation import train_test_split  #the moduel that can split the training set into diffrent groups

Xd_train,Xd_test,y_train,y_test=train_test_split(X_d,y,random_state=14)#cut the set ramdomly for testing and training

if __name__ == "__main__":
    all_predictors={}
    errors={}
    for feature_index in range(Xd_train.shape[1]):#shape[1] gets the size of the training array?
        # print(feature_index)#0,1,2,3
        predictors,total_error=train_on_feature(Xd_train,y_train,feature_index)
        print(train_on_feature(Xd_train,y_train,feature_index))
        all_predictors[feature_index]=predictors
        errors[feature_index]=total_error


        #next step is to find out the best rule
        best_feature,best_error=sorted(errors.items(),key=itemgetter(1))[0]#the best choise is at the front
        # print("best_feature: {0}".format(best_feature))
        # print("best_error: {0}".format(best_error))

        #next, we are going to test the acurrence
        model={'feature':best_feature,'predictor':all_predictors[best_feature][0]}
        y_predicted=predict(Xd_test,model)
        accuracy=np.mean(y_predicted==y_test)*100
        print("The test accuracy is {:.if}%".format(accuracy))
