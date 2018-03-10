from numpy import*
import operator

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify(testX,dataSet,K):
    dataSet, labels = createDataSet()
    testX = [0, 0]
    # get the size for dataSet
    dataSetSize = dataSet.shape[0]
    # create a matrix for testX
    mat = tile(testX, (dataSetSize, 1))
    # calculate (x1-x2),(y1-y2)
    subtrac = mat - dataSet
    squareSubtrac = subtrac ** 2
    # sum (x1-x2)^2 and (y1-y2)^2
    squareDistance = squareSubtrac.sum(axis=1)
    # sqrt((x1-x2)^2 + (y1-y2)^2)
    distance = squareDistance ** 0.5
    # sort the value of distance and return the index
    sortedDistance = distance.argsort()
    ClassCount = {}
    for i in range(K):
        voteLabel = labels[sortedDistance[i]]
        # count the amount for each label and store it to ClassCount
        ClassCount[voteLabel] = ClassCount.get(voteLabel,
                                               0) + 1  # only ".get(,0)"can be used for the reference before initialized
    sortedClassCount = sorted(ClassCount.items(), key=operator.itemgetter(1), reverse=True)
    print(sortedClassCount)
    print("New data belongs to label "+str(sortedClassCount[0][0]))


dataSet,labels=createDataSet()
#test weather [0,0] belongs to label A(or B)
testX=[0,0]
classify(testX,dataSet,3)

''''
#tile(vec,(x,y)) :repeat vec for x times--x rows and y times for each row
diffMat=tile([0,0],(dataSetSize,1))-dataSet
sqrDiffMat=diffMat**2
sqrDistances=sqrDiffMat.sum(axis=1)#axis:轴线
distance=sqrDistances**0.5
sortDistIndicate=distance.argsort()#return the index of values stored from small to greatargsort函数返回的是数组值从小到大的索引值[3, 1, 2]从小到大为[1，2，3],期对应的索引为[1，2，0]
classCount={}
for i in range(3):
    voteLabel=labels[sortDistIndicate[i]]
    classCount[voteLabel]=classCount.get(voteLabel,0)+1#only ".get(,0)"can be used for the reference before initialized
    print(voteLabel+": "+str(classCount[voteLabel]))
sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
print(sortedClassCount)

'''