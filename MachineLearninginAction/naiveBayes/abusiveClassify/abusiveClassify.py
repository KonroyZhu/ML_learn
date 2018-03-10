from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec
# dataSet,classVec=loadDataSet()
# # print(dataSet)

#founction to get all the vocabularies from data set
#remove duplicate
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        #combine vocabSet and set(document) ---similar to append() for list
        vocabSet=vocabSet|set(document)#"|" can combine the set from both side
    return list(vocabSet)

# vocabSet=createVocabList(dataSet)
# print(vocabSet)

#founction to transform dataSet into vector
def setOfWordVec(vocabSet,featSet):
    returnVec=[0]*len(vocabSet)
    for vocab in vocabSet:
        if vocab in featSet:
            returnVec[vocabSet.index(vocab)]=1
        else:
            returnVec[vocabSet.index(vocab)]=0
    return returnVec



# print(trainMat)
def trainNB0(trainMat,listClasses):

    #find out how many training sentence is in the set
    numTrain=len(trainMat)
    #how many word is in a vector
    numOfWords=len(trainMat[0])
    #find out the possibility of the abusive sentence in the training set
    pAbusive=float(sum(listClasses))/numTrain

    #build a vector to hold possiblity for 1(abusive) and 0(non-abusive)
    # p0Num=zeros(numOfWords);p1Num=zeros(numOfWords)
    # p0Denominator=0.0;p1Denominator=0.0
    ####to avoid the situation that the word doesn't appear which lead to p=0, we have to smooth the data by the follwing modify
    p0Num=ones(numOfWords);p1Num=ones(numOfWords)
    p0Denominator=2.0;p1Denominator=2.0
    ####
    for i in range(numTrain):
        if listClasses[i]==1:
            #it's an abusive sentence
            #add the value from trainMat[i] to vector p1Num
            p1Num+=trainMat[i]#vertically
            p1Denominator+=sum(trainMat[i])#travsely
        else:
            #it's not an abusive sentence
            p0Num+=trainMat[i]
            p0Denominator+=sum(trainMat[i])
    # p1Vec=p1Num/p1Denominator#the possibility that certain vocabulary will appear in an abusive sentence
    # p0Vec=p0Num/p0Denominator#the possibility that certain vocabulary(in side the vector) will appear in a non-abusive sentence
    ####to avoid underflow and too much of multiplication of small number, we do the following modify
    p1Vec=log(p1Num/p1Denominator)
    p0Vec=log(p0Num/p0Denominator)
    return p1Vec,p0Vec,pAbusive


listOPost, listClasses = loadDataSet()
vocabList = createVocabList(listOPost)
trainMat = []
for postinDoc in listOPost:
    # print(postinDoc)
    print(setOfWordVec(vocabList, postinDoc))
    trainMat.append(setOfWordVec(vocabList, postinDoc))
p1Vec,p0Vec,pAbusive=trainNB0(trainMat,listClasses)
print("the possiblity of abusive sentences.txt in the trainSet:"+str(pAbusive))
print("the possibility that certain vocabulary will appear in an abusive sentences.txt:"+"\n"+str(p1Vec))
print("the possibility that certain vocabulary will appear in a non-abusive sentences.txt:"+"\n"+str(p0Vec))



def classify(testVect,p1Vect,p0Vect,pClass):
    p1=sum(testVect*p1Vec)+log(pClass)
    p0=sum(testVect*p0Vec)+log(1.0-pClass)
    if p1>p0:
        return print("yes")
    else:
        return print("no")


testList=['stupid']
testVect=setOfWordVec(vocabList,testList)
# print(testVect)
classify(testVect,p1Vec,p0Vec,pAbusive)