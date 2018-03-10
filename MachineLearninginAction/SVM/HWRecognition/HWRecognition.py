from numpy import *

def img2vector(filename):
    returnVec=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVec[0,i*32+j]=int(lineStr[j])
    return returnVec

def loadImage(dirName):
    from os import listdir
    hwLabels=[]
    trainingFileList=listdir(dirName)
    #the number of training files
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])

        #our task is to recognize the number 9
        if classNumStr==9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i,:]=img2vector(str(dirName)+"/"+str(fileNameStr))
    return trainingMat,hwLabels


def testDigits(kTup=('rbf',10)):
    dataArr,labelArr=loadImage('trainingDights')
    b,alphas=somP(dataArr,labelArr,200,0.0001,10000,kTup)
    dataMat=mat(dataArr);labelMat=mat(labelArr).transpose()

    #index of support vectors
    svInd=nonzero(alphas.A>0)[0]
    sVs=dataMat[svInd]
    labelSV=labelMat[svInd]
    print("there are "+str(shape(sVs)[0])+" Support Vectors")

    m,n=shape(dataMat)
    errorCount=0
    for i in range(m):
        kernelEval=kernelTrans(sVs,dataMat[i,:],kTup)
        predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b

























