#以矩阵形式创建数据集
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
	    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
	    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
	    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
	    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
	    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'],
        ['you','stupid','man']]
    classVec = [0,1,0,1,0,1,1]    #1 is abusive, 0 not
    return postingList,classVec
data,label=loadDataSet()
# print(len(data))

#将矩阵内容添加到列表，set获取list中不重复的元素
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        # print(document)
        vocabSet = vocabSet | set(document) #union of the two sets

    return list(vocabSet)

words_bag=createVocabList(data)
# print(words_bag)

#判断list中每个词在总共词语list中的位置
def setOfWords2Vec(words_bag, data):
    returnVec = [0]*len(words_bag)
    for word in data:
        if word in words_bag:
            returnVec[words_bag.index(word)] = 1
        else: print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec


import numpy as np
from sklearn.naive_bayes import GaussianNB

dataX=data
trainX=[]
for line in dataX:
    trainX.append(setOfWords2Vec(words_bag,line))

X=np.array(trainX[:5])
Y=np.array(label[:5])

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X,Y)
predicted=model.predict(np.array(trainX[6]))
for p in predicted:
    print(p)
