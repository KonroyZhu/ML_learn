


#数据加载
import numpy as np
def load_data_with_numpy(path):
    dataset=np.loadtxt(path,delimiter=",")
    # print(len(dataset[0]))
    X=dataset[:700,:8]#get all rows from dataset by " : "  get 0~8 column by "0:8"

    # print(len(X[0]))
    y=dataset[:700,-1]#get all rows from dataset by ":"   get the last column witch represent the class of the row divided by 0 or 1 by "-1"
    # print(len(y))

    TestX=dataset[700:,:8]
    Testy=dataset[700:,-1]
    return dataset,X,y,TestX,Testy


dataset,X,y,TestX,Testy=load_data_with_numpy("./dataset")
# print(len(dataset[0]))
# print(len(X[0]))#data
# print(y)#labe



# 数据标准化
from sklearn import preprocessing
def normalize_data_with_sklearn(dataset):
    normalized_dataset=preprocessing.normalize(dataset)#make sure all data attribute are between 0~1
    # print(normalized_X[0])
    standardized_dataset=preprocessing.scale(dataset)#negative value is included?
    # print(standardized_X)
    return normalized_dataset,standardized_dataset

normalized_dataset,standardized_dataset=normalize_data_with_sklearn(X)

# 特征的选取 method1
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
def check_features_importance_by_TreeClassifier():
    model=ExtraTreesClassifier()
    model.fit(X,y)
    features_importance=model.feature_importances_
    return features_importance

features_importance=check_features_importance_by_TreeClassifier()
# print(features_importance)#the importance value for each column from 0 to 7




#we can choose the best value with the largest entropy frlm features_importance

# 特征的选取 method2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
rfe=RFE(model,3)
rfe=rfe.fit(X,y)
# print(rfe.support_)
print("method2 ranking: ")
print(rfe.ranking_)

#some conflic occur in method1 and method2


#算法的开发
# 支持向量机
# SVM（支持向量机）是最流行的机器学习算法之一，它主要用于分类问题。同样也用于逻辑回归，SVM在一对多方法的帮助下可以实现多类分类。
from sklearn.svm import SVC
def classify_by_SVM(FeatX,Featy,TestX,Testy):
    model = SVC()
    model.fit(FeatX, Featy)

    hit = 0
    expected = Testy
    predicted = model.predict(TestX)
    for i in range(len(predicted)):
        if predicted[i] == expected[i]:
            hit += 1
    print("SVM hit points presentage: " + str(hit / len(predicted)))

classify_by_SVM(X,y,TestX,Testy)

# 逻辑回归
# 大多数情况下被用来解决分类问题（二元分类），但多类的分类（所谓的一对多方法）也适用。这个算法的优点是对于每一个输出的对象都有一个对应类别的概率。

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
def classify_by_LogisticRegression(FeatX,Featy,TestX,Testy):
    model = LogisticRegression()
    model.fit(FeatX, Featy)
    # print(softmax_model)
    # make predictions
    expected = Testy
    predicted = model.predict(TestX)
    # summarize the fit of the softmax_model
    # print(expected)
    # print(predicted)
    hit=0
    for i in range(len(predicted)):
        if predicted[i]==expected[i]:
            hit+=1
    print(" LogisticRegression hit points presentage: "+str(hit/len(predicted)))


classify_by_LogisticRegression(X,y,TestX,Testy)
# print(X)
# print(TestX)

# 朴素贝叶斯
# 它也是最有名的机器学习的算法之一，它的主要任务是恢复训练样本的数据分布密度。这个方法通常在多类的分类问题上表现的很好。
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
def classify_by_GaussianNB(FeatX,Featy,TestX,Testy):
    model = GaussianNB()
    model.fit(FeatX,Featy)

    hit=0
    expected=Testy
    predicted=model.predict(TestX)
    for i in range(len(predicted)):
        if predicted[i]==expected[i]:
            hit+=1
    print("GaussianNB  hit points presentage: "+str(hit/len(predicted)))

classify_by_GaussianNB(X,y,TestX,Testy)


# k-最近邻
# kNN（k-最近邻）方法通常用于一个更复杂分类算法的一部分。例如，我们可以用它的估计值做为一个对象的特征。有时候，一个简单的kNN算法在良好选择的特征上会有很出色的表现。当参数（主要是metrics）被设置得当，这个算法在回归问题中通常表现出最好的质量。
from sklearn.neighbors import KNeighborsClassifier
def classify_by_KNNClassifier(FeatX,Featy,TestX,Testy):
    model=KNeighborsClassifier()
    model.fit(FeatX, Featy)

    hit = 0
    expected = Testy
    predicted = model.predict(TestX)
    for i in range(len(predicted)):
        if predicted[i] == expected[i]:
            hit += 1
    print("KNN  hit points presentage: " + str(hit / len(predicted)))

classify_by_KNNClassifier(X,y,TestX,Testy)

# 决策树
# 分类和回归树（CART）经常被用于这么一类问题，在这类问题中对象有可分类的特征且被用于回归和分类问题。决策树很适用于多类分类。
from sklearn.tree import DecisionTreeClassifier
def classify_by_decisionTree(FeatX,Featy,TestX,Testy):
    model = DecisionTreeClassifier()
    model.fit(FeatX, Featy)

    hit = 0
    expected = Testy
    predicted = model.predict(TestX)
    for i in range(len(predicted)):
        if predicted[i] == expected[i]:
            hit += 1
    print("Claasify tree  hit points presentage: " + str(hit / len(predicted)))

classify_by_decisionTree(X,y,TestX,Testy)

