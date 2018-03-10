
from numpy import *
from math import log

def createVocabSet(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    return vocabSet

def setOfWordVec(vocabSet,posInDoc):
    returnVec=[0]*len(vocabSet)
    for vocab in vocabSet:
        if vocab in posInDoc:
            returnVec[vocabSet.index(vocab)]=1
    return returnVec

def trainNB0(trainMat,listClass):
    numOfTrainDoc=len(trainMat)
    numOfWords=len(trainMat[0])
    pSpam=float(sum(listClass))/numOfTrainDoc

    p0Num=ones(numOfWords);p1Num=ones(numOfWords)
    p0Denominator=2.0;p1Denominator=2.0

    for i in range(numOfTrainDoc):
        if listClass[i]==1:
            p1Num+=trainMat[i]
            p1Denominator+=sum(trainMat[i])
        else:
            p0Num+=trainMat[i]
            p0Denominator+=sum(trainMat[i])

    p1Vec=log(p1Num/p1Denominator)
    p0Vec=log(p0Num/p0Denominator)
    return pSpam,p1Vec,p0Vec

def classifyNB(testVec,p0Vec,p1Vec,pSpam):
    p1=testVec*p1Vec+log(pSpam)
    p0=testVec*p0Vec+log(1-pSpam)
    if p0>p1:
        return 0
    else:
        return 1


# print(open('email/ham/6.txt').read())#UnicodeDecodeError
# print(open('email/ham/6.txt',encoding='utf-8', errors='ignore').read())
def testParse(bigString):
    import  re
    listOfToken=re.split(r'\W*',bigString)
    return [token for token in listOfToken if len(token)>2]


def spamTest():
    docList=[]
    classList=[]

    for i in range(26):
        wordList=testParse(open('email/ham/6.txt',encoding='utf-8', errors='ignore').read())
        docList.append(wordList)
        classList.append(0)
        wordList=testParse(open('email/spam/6.txt',encoding='utf-8', errors='ignore').read())
        docList.append(wordList)
        classList.append(1)
    vocabSet=createVocabSet(docList)
    trainSet=list(range(50));testSet=[]
    #code inside the for loop creates the index radomly in testSet(and the rest are stored in trainSet)
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainSet)))
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])

    trainMat=[];trainClass=[]
    for docIndex in trainSet:
        trainMat.append(setOfWordVec(vocabSet,docList[docIndex]))
        trainClass.append(classList[docIndex])
    pSpam,p1Vec,p0Vec=trainNB0(trainMat,trainClass)


    ###test
    errorCount=0.0
    for docIndex in testSet:
        wordVect=setOfWordVec(vocabSet,docList[docIndex])
        if classifyNB(wordVect,p0Vec,p1Vec,pSpam)!=trainClass[docIndex]:
            errorCount+=1
    print("the error rate is : "+str(float(errorCount)/len(testSet)))



spamTest()
# trainSet=list(range(50));testSet=[]
# for i in range(10):
#     randIndex = int(random.uniform(0, len(trainSet)))
#     testSet.append(trainSet[randIndex])
#     del (trainSet[randIndex])
# print(testSet)
# print(trainSet)