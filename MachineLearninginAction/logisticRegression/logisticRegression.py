from math import exp
from numpy import*


def loadDataSet():
    dataMat=[];labelMat=[]
    fr=open('testSet.txt')
    for lineArry in fr.readlines():
        lineArry=lineArry.strip().split()
        # print(lineArry[2])
        dataMat.append([1.0,float(lineArry[0]),float(lineArry[1])])
        labelMat.append(lineArry[2])
    return dataMat,labelMat

def sigmoid(inX):
    #exp(x)=e^x
    #return the value between 0 and 1 (sigmoid(0)=0.5 neutral)
    return 1.0/(1+exp(-inX))


# dataMat,labelMat=loadDataSet()
def gradAscent(dataMat,labelMat):
    dataMatrix=mat(dataMat)
    labelMat=mat(labelMat).transpose()
    m,n=shape(dataMatrix)#(100*3)
    alpha=0.001
    maxCycles=500
    weights=ones((n,1))#(3*1)

    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)#(100*1)
        errors=(labelMat.astype('float64')-h)#(100*1)##http://www.th7.cn/Program/Python/201606/870193.shtml
        weights=weights+alpha*dataMatrix.transpose()*errors#(3*100)*(100*3)
    return weights

#IMPROVE1
#the founction gradAscent is unnecessary expensive in computation
def stochGradAscent(dataMat,classLabels):
    m,n=shape(dataMat)#100,3
    alpha=0.01
    weights=ones(n)#1*3
    for i in range(m):
        h=sigmoid(sum(dataMat[i]*weights))#turn 3 value into 1
        error=float(classLabels[i])-h
        weights=weights+alpha*error*array(dataMat[i])#update weights by each training data
    return mat(weights).transpose()

#IMPROVE2
#update weight with the vector selected randomly
#alpha changes on each iteration
def imStochGradAscent(dataMat,classLabels,iterTimes):
    m,n=shape(dataMat)
    weights=ones(n)
    for j in range(iterTimes):
        dataIndex=range(m)
        for i in range(m):
            alpha=4/(1.0+i+j)+0.01
            randIndex=int(random.uniform(0,len(dataMat)))
            h=sigmoid(sum(dataMat[i]*weights))
            error=float(classLabels[i])-h
            weights=weights+error*alpha*array(dataMat[i])

    return mat(weights).transpose()






def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weight=wei.getA()
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=shape(dataMat)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord2, ycord2, s=30, c='blue')
    ax.scatter(xcord1,ycord1,s=30,c='red')

    x=arange(-3.0,3.0,0.1)
    y=(-weight[0]-weight[1]*x)/weight[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

dataMat,labelMat=loadDataSet()

#1
weights1=gradAscent(dataMat,labelMat)
# print(weights1)
# plotBestFit(weights1)
#2
weights2=stochGradAscent(dataMat,labelMat)
# print(weights2)
# plotBestFit(weights2)
#3
weight3=imStochGradAscent(dataMat,labelMat,150)
plotBestFit(weight3)







