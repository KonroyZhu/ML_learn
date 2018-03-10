from math import log
import operator

#Create dataSet
def createDataSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels
##______________________________________________________________________________________________________________________



#Create the tree
#information gain
def calcShannonEnt(dataSet):
    #H= p(x1)*log2p(x1)+p(x2)*log2p(x2)+...+p(xn)*log2p(xn)
    numEntries=len(dataSet)
    labelCount={}
    for featVect in dataSet:
        currentLabel=featVect[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel]=0
        labelCount[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCount:
        prob=float(labelCount[key])/numEntries
        shannonEnt-=prob*log(prob,2)
    return shannonEnt


# myDat,labels=createDataSet()
# print(myDat)
# print(calcShannonEnt(myDat))
##change myDat
# myDat[0][-1]='maybe'
# print(myDat)
# print(calcShannonEnt(myDat))

#drop the value inside the vector vertically
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        # print("featVect:")
        # print(featVec)
        if featVec[axis]==value:
            #if the axis'th element in featVect meets the condition drop the value and add the vector to retDataSet
            retFeatVec=featVec[:axis]
            retFeatVec.extend(featVec[axis+1:])#featVec[axis] is dropped
            retDataSet.append(retFeatVec)
    return  retDataSet

#find the best feature to split on with the founction splitDataSet() and calcShannonEnt()
def chooseBestFeatToSplit(dataSet):
    numberOfFeat=len(dataSet[0])-1#the last one is the label
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain=0.0
    bestFeat=-1

    #try all features
    for i in range(numberOfFeat):
        #get teh features vertically
        featList=[example[i] for example in dataSet]
        #get all possible value from features
        uniqueVals=set(featList)#element in a set can not be duplicated

        newEntropy=0.0
        #try all feature values
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            p=len(subDataSet)/float(len(dataSet))
            newEntropy+=p*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if(bestInfoGain<infoGain):
            bestInfoGain=infoGain
            bestFeat=i
    return bestFeat

#get the most frequency appeared label in classList
def majorityCount(ClassList):
    ClassCount={}
    for key in ClassList:
        if key not in ClassCount.keys():
            ClassCount[key]=0
        ClassCount[key]+=1
    sortedClassCount=sorted(iterable=ClassCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]



def createTree(dataSet,label):#label contain the name of the feature('surfacing''filpper')
    #classList store the class('yes' or 'no')
    classList=[example[-1] for example in dataSet]
    #to check if all the classes are the same (when more and more features are splitted it tend more to be the same
    #(eg:classList.count('yes') returns how many 'yes' are included in the classList)
    if classList.count(classList[0])==len(classList):
        return classList[0]
    #to check if all the features has been splited on
    #that is to say dataSet has no feature left but the class('yes' or 'no')
    if len(dataSet[0])==1:
        return majorityCount(classList)

    #choose the bestFeat to split on and build the tree
    bestFeat=chooseBestFeatToSplit(dataSet)
    bestFeatLabel=label[bestFeat]

    #use data structure dictionary to represent the tree
    myTree={bestFeatLabel:{}}
    #delete the label('surfacing' or 'flipper')
    del(label[bestFeat])

    #try out all the possible value in feature
    featValues=[example[bestFeat] for example in dataSet]
    #remove duplicate
    uniqueValues=set(featValues)
    for value in uniqueValues:
        #subLabel is the label list without the bestFeatLabel
        subLabel=label[:]
        #myTree[bestFeatLabel][value] is a dictionary(value) inside the dictionary(bestFeatLabel) which represents the different branch of the tree
        #insert a dictionary into the dictionary
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabel)

    return myTree





# myData,labels=createDataSet()
# print("labels: "+str(labels))
# myTree=createTree(myData,labels)
# print("myTree:")
# print(myTree)



##______________________________________________________________________________________________________________________

#Using the tree as modules

myData,labels=createDataSet()
# print("labels: "+str(labels))
myTree=createTree(myData,labels)
myData,labels=createDataSet()

#inputTree for myTree , FeatLabel for labels
def classify(inputTree,FeatLabel,testVec):

    firstStr=list(inputTree.keys())[0]
    secondDict=inputTree[firstStr]
    labelIndex=FeatLabel.index(firstStr)

    #go throught all the possible child
    for key in secondDict.keys():
        classLabel=""
        #value from the testVector equal this branch
        if  testVec[labelIndex]==key:
            #when it wasn't a leaf node then continue to go throught the dict
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],FeatLabel,testVec)
            #if it hit a leaf node then return the class
            else:
                classLabel=secondDict[key]
        return classLabel


print(classify(myTree,labels,[0,1]))



##______________________________________________________________________________________________________________________
#Plotting myTree_
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


#to plotin myTree we need to know the depth and amount of the leaf
# print(myTree.keys()[0])#TypeError: 'dict_keys' object does not support indexing
# print(list(myTree.keys())[0])#solution
def getNumLeafs(myTree):
    numberOfLeafs=0
    # get the first key from the dictionary
    firstStr=list(myTree.keys())[0]
    # get the value by using the key
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        # check the type of the node
        #if the type is dictionary
        if type(secondDict[key]).__name__=='dict':
            # do the recursion to count the leafs in next dictionary
            numberOfLeafs+=getNumLeafs(secondDict[key])
        # if it's not a dictionary but a leafNode at the end
        else:
            numberOfLeafs+=1
    return numberOfLeafs

def getTreeDepth(myTree):
    maxTreeDepth=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in  secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else:
            thisDepth=1
        #count the depth of the tree when it hits the leaf node and find out the largest one
        if thisDepth>maxTreeDepth:
            maxTreeDepth=thisDepth
    return maxTreeDepth

# print("number of leaf nodes: "+str(getNumLeafs(myTree)))
# print("number of levels: "+str(getTreeDepth(myTree)))



def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):  # if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  # this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]  # the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) /(2.0 *plotTree.totalW) , plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD#change the yOff after plotting each node


# if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))#globel variable1 to store Width
    plotTree.totalD = float(getTreeDepth(inTree))#globel variabel to store Depth
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


# createPlot(myTree)

##______________________________________________________________________________________________________________________























'''
print(myDat)
numOfFeatures=len(myDat[0])-1#the last element is the label
baseEntropy=calcShannonEnt(myDat)#initialize
bestInfoGain=0.0
bestFeat=-1
for i in range(numOfFeatures):
    featureList=[example[i] for example in myDat]
    # print(featureList)
    uniqueVals=set(featureList)#Duplicate removal by set()//data inside a set can't be the same
    # print(uniqueVals)
    newEntropy=0.0
    for value in uniqueVals:
        #try to split the dataSet on each feature and calculate it's entropy to choose the best one
        subDataSet=splitDataSet(myDat,i,value)
        p=len(subDataSet)/float(len(myDat))
        newEntropy+=p*calcShannonEnt(subDataSet)
    inforgain=baseEntropy-newEntropy
    if(inforgain>bestInfoGain):
        bestInfoGain=inforgain
        bestFeat=i
print(bestFeat)
'''
''''
def getNumberOfLeafs(MyTree):
 numberOfLeafs = 0
 # get the first key from the dictionary
 firstStr = list(myTree.keys())[0]
 # get the value by using the key
 secondDict = myTree[firstStr]
 for key in secondDict.keys():
     # check the type of the node
     # if the type is dictionary
     if type(secondDict[key]).__name__ == 'dict':
         # do the recursion to count the leafs in next dictionary
         numberOfLeafs += getNumberOfLeafs(secondDict[key])
     # if it's not a dictionary but a leafNode at the end
     else:
         numberOfLeafs += 1
 return numberOfLeafs
'''