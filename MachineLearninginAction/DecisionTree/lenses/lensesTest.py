

#CREATE DATASET
f=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in f.readlines()]
# print(lenses[0])

lensesLabel=['age','prescript','astigmatic','tearRate']

#BUILDING DECISION TREE
#founction to calculate the entropy
def calEntropy(dataSet):
    from math import log
    # H= p(x1)*log2p(x1)+p(x2)*log2p(x2)+...+p(xn)*log2p(xn)
    numEnt=len(dataSet)
    classCount={}
    for featVec in dataSet:
        classLabel=featVec[-1]
        if classLabel not in classCount.keys():
            classCount[classLabel]=0
        classCount[classLabel]+=1
    entropy=0.0
    for key in classCount:
        prop=float(classCount[key])/numEnt
        entropy-=prop*log(prop,2)
    return entropy
#founction to split the dataSet on certain feat with certain value
def splitDataSet(dataSet,axis,value):
    reDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            retFeatVec=featVec[:axis]
            retFeatVec.extend(featVec[axis+1:])
            reDataSet.append(retFeatVec)
    return reDataSet
#founction to choose best feature to split on
def chooseBestFeatToSplit(dataSet):
    numberOfFeat=len(dataSet[0])-1#the last one is the label
    baseEntropy=calEntropy(dataSet)
    bestInfoGain=0.0
    bestFeat=-1

    for i in range(numberOfFeat):
        classLabel=(example[-1] for example in dataSet)
        uniqueClassLabel=set(classLabel)

        newEntropy=0.0
        for value in uniqueClassLabel:
            subSet=splitDataSet(dataSet=dataSet,axis=i,value=value)
            p=float(len(subSet))/float(len(dataSet))
            newEntropy+=p*calEntropy(subSet)
        newInfoGain=baseEntropy-newEntropy
        if bestInfoGain<newInfoGain:
            bestInfoGain=newInfoGain
            bestFeat=i
    return bestFeat
#founction to find the major label
# def majorityCount(classList):
#     import operator
#     classCount={}
#     for classLabel in classList:
#         if classLabel not in classCount.keys():
#             classCount[classLabel]=0
#         classCount[classLabel]+=1
#     sortedClassCount=sorted(iterable=classCount.items(),key=operator.itemgetter(1),reverse=True)
#     return sortedClassCount[0][0]

def majorityCount(ClassList):
    import operator
    ClassCount={}
    for key in ClassList:
        if key not in ClassCount.keys():
            ClassCount[key]=0
        ClassCount[key]+=1
    sortedClassCount=sorted(iterable=ClassCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


#main founction
#founction to create the decision tree
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]

    if classList.count(classList[0])==len(classList):
        #when all class is the same
        return classList[0]
    if len(dataSet[0])==1:
        #when there's nothing left except for the classLabel
        return majorityCount(classList)


    bestFeat=chooseBestFeatToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    DecTree={bestFeatLabel:{}}
    del(labels[bestFeat])

    featValues=[example[bestFeat] for example in dataSet]
    uniqueValues=set(featValues)
    for values in uniqueValues:
        subLabels=labels[:]
        DecTree[bestFeatLabel][values]=createTree(splitDataSet(dataSet=dataSet,axis=bestFeat,value=values),subLabels)

    return DecTree


decisionTree=createTree(lenses,lensesLabel)
# print(decisionTree)
lensesLabel=['age','prescript','astigmatic','tearRate']


def classify(trainedTree,labels,testVec):
    firstStr=list(trainedTree.keys())[0]
    secondDist=trainedTree[firstStr]
    labelIndex=labels.index(firstStr)

    for key in secondDist.keys():

        if testVec[labelIndex]==key:
            if type(secondDist[key]).__name__=='dict':
                classLabel=classify(secondDist[key],labels,testVec)
            else:
                classLabel=secondDist[key]
    return  classLabel

testVect = ['presbyopic', 'hyper', 'yes', 'normal']
print(classify(decisionTree,lensesLabel,testVect))