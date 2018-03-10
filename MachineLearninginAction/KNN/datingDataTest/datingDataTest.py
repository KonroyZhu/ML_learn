from numpy import*
import  matplotlib.pyplot as plt
import operator

def fileFromat(filename):

    fr=open(filename,'r')
    numberOflines=len(fr.readlines())
    #create a matrix to store the data( 3 features in each line for this case)
    datingMat=zeros((numberOflines,3))
    #create a list to save classify labels
    datingLab=[]
    fr=open(filename,'r')
    index=0
    for line in fr.readlines():
        line=line.strip()
        #split the data by space('\t')
        line=line.split('\t')
        #store data
        #0,1,2 refers to 3 features
        datingMat[index,:]=line[0:3]#with 3 excluded
        #3 refers to the label
        datingLab.append(int(line[-1]))
        index+=1

    return datingMat,datingLab

#1 get formated data
datingMat, datingLab = fileFromat("datingData")
# print(datingMat)
#----------------------------------------------------------------------------------------------------------------------#

def view(Feat):

    x=Feat[:,1]
    y=Feat[:,0]
    colors=array(datingLab)
    # color array and x,y should be in the same size
    # otherwise exception: "color array should be two-demsion" will be called
    print("x size: " + str(x.size))
    print("y size:" + str(y.size))
    print("colors size:" + str(colors.size))
    plt.scatter(x, y, c=colors)
    plt.show()

#2 view the data
# view(datingMat)
#----------------------------------------------------------------------------------------------------------------------#


def autoNormal(Feat):
    #normalizedValue=((oldvalue-minValue)/(maxValue-minValue))
    minValue=Feat.min(0)
    maxValue=Feat.max(0)
    denominator=maxValue-minValue#a vector with 3 values

    #create NormlFeat to store nomalized data
    normalizedMat=zeros(Feat.shape)
    m= Feat.shape[0]  # Feat.shape=(1000,3)
    #oldvalue-minValue)
    normalizedMat=Feat-tile(minValue,(m,1))#tile(vector,(x,1)): repeat vector for x times in vertical dimension
    #/(maxValue-minValue)
    normalizedMat=normalizedMat/tile(denominator,(m,1))
    normalizedMat*=10

    return normalizedMat,minValue,denominator#return minValue and range to normalized the test data

#3 nomalize the data
normalizedMat,minValue,denominator=autoNormal(datingMat)
print(normalizedMat)
# view(normalizedMat)
#----------------------------------------------------------------------------------------------------------------------#


def classify(testX,Mat,Label,K):
    # get the size for dataSet
    dataSetSize = Mat.shape[0]
    # create a matrix for testX
    mat = tile(testX, (dataSetSize, 1))
    # calculate (x1-x2),(y1-y2)
    subtrac = mat - Mat
    squareSubtrac = subtrac ** 2
    # sum (x1-x2)^2 and (y1-y2)^2
    squareDistance = squareSubtrac.sum(axis=1)
    # sqrt((x1-x2)^2 + (y1-y2)^2)
    distance = squareDistance ** 0.5
    # sort the value of distance and return the index
    sortedDistance = distance.argsort()
    ClassCount = {}
    for i in range(K):
        voteLabel = datingLab[sortedDistance[i]]
        # count the amount for each label and store it to ClassCount
        ClassCount[voteLabel] = ClassCount.get(voteLabel,
                                               0) + 1  # only ".get(,0)"can be used for the reference before initialized
    sortedClassCount = sorted(ClassCount.items(), key=operator.itemgetter(1), reverse=True)
    # print(sortedClassCount)
    # print("New data belongs to label " + str(sortedClassCount[0][0]))
    return sortedClassCount[0][0]


#4 KNN classify
# datingMat, datingLab = fileFromat("datingData")
# normalMat, minValue, denominator = autoNormal(datingMat)
# classify(testX=[ 0.44832535 , 0.39805139  ,0.56233353],Mat=normalMat,Label=datingLab,K=3)
#----------------------------------------------------------------------------------------------------------------------#




#
datingMat,datingLab=fileFromat("datingData")
# view(datingMat)
normalMat,minValue,denominator=autoNormal(datingMat)
testRatio=0.1#10% vectors of normalMat are for testing
testVectorNumber=int(normalMat.shape[0]*testRatio)

errorTimes=0.0
for i in range(testVectorNumber):
    classifyResult=classify(testX=datingMat[i,:],Mat=datingMat[testVectorNumber:datingMat.shape[0],:],Label=datingLab,K=5)
    print("classify result: "+str(classifyResult)+" it should be: "+str(datingLab[i]))
    if(classifyResult!=datingLab[i]):
        errorTimes+=1.0
print("the total error rate is: "+str(errorTimes/testVectorNumber))