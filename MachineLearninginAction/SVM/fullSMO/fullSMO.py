from numpy import *

def loadDataSet(filename):
    dataMat=[];labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat


##use kernels for more complex data
def kernelTrans(X,A,kTup):
    m,n=shape(X)
    K=mat(zeros((m,1)))
    if kTup[0]=='lin':#linear kernels
        K=X*A.T#full dataSet and a row of the dataSet
    elif kTup[0]=='rbf':#radial bias function
        for j in range(m):
            deltaRow=X[j,:]-A
            K[j]=deltaRow*deltaRow.T
        K=exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('Kernel is not recongnized')
    return K
##use kernels for more complex data



#used as a data structure to hold all the important imfomation
class optStruct:
    #initialize
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=shape(dataMatIn)[0]
        self.alphas=mat(zeros((self.m,1)))
        self.b=0
        #the first element in eCache is a flag stating whether eCache is valid, the second one is the value
        self.eCache=mat(zeros((self.m,2)))#error cache

        ##use kernels for more complex data
        self.K=mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.X,self.X[i,:],kTup)


#function to calculating the error for the given alpha k
#oS is an instance of optStruct; k is an index
def calcEk(oS,k):
    # fXk=float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,: ].T))+oS.b
    ##use kernels for more complex data
    fXk=float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k]+oS.b)
    Ek=fXk-float(oS.labelMat[k])
    return Ek

#function to select index j for alpha j randomly
def selectJrand(i,m):
    #i is the index of our first alpha
    #m is the total number of alpha
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j

#clip alpha values that are greater than H or less than L
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if aj<L:
        aj=L

    return aj

#function to select alpha[j] base on alpha[i],instance of optStruct and error value associ]ate with alpha[i]
#select a second alpha so that we took the max step in each optimize
def selectJ(i,oS,Ei):
    maxK=-1;maxDeltaE=0;Ej=0
    oS.eCache[i]=[1,Ei]#initialize the eCache set the validate flag to 1 which means it has been calculated
    # nonzeros() returns the index of element whose value!=0
    #oS.eCache[:,0] returns every value from the first column
    #validEcacheList hold the alpha corresponding with non-zero E
    validEcacheList=nonzero(oS.eCache[:,0].A)[0]
    if(len(validEcacheList))>1:
        #if the flag show that the eCache is valid
        for k in validEcacheList:
            if k==i :
                continue
            Ek=calcEk(oS,k)
            deltaE=abs(Ei-Ek)
            if(deltaE>maxDeltaE):
                maxK=k;maxDeltaE=deltaE;Ej=Ek
        return maxK,Ej
    else:
        #select a j randomly
        j=selectJrand(i,oS.m)
        Ej=calcEk(oS,j)
    return j,Ej

#function to update by calculating the error and put it into eCache
def updateEk(oS,k):
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1,Ek]

#inner loop
def innerL(i,oS):
    Ei=calcEk(oS,i)
    #to see if alphas[i] is validated
    if ((oS.labelMat[i]*Ei<-oS.tol)and(oS.alphas[i]<oS.C))   or   ((oS.labelMat[i]*Ei>oS.tol)and(oS.alphas[i]>0)):
        #when i is ok then select j(the second alpha)
        j,Ej=selectJ(i,oS,Ei)

        #mark down the old j and old i
        alphaIold=oS.alphas[i].copy();alphaJold=oS.alphas[j].copy()

        if(oS.labelMat[i]!=oS.labelMat[j]):
            L=max(0,oS.alphas[j]-oS.alphas[i])
            H=min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
           L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
           H=min(oS.C,oS.alphas[j]+oS.alphas[i])
        if (L==H):
            print("L==H")
            return 0

        # eta=2.0*oS.X[i,:]*oS.X[j,:].T-oS.X[i,:]*oS.X[i,:].T-oS.X[j,:]*oS.X[j,:].T
        ##use kernels for more complex data
        eta=2.0*oS.K[i,j]-oS.K[i,i]-oS.K[j,j]
        if eta>=0:
           print("eta>=0");return 0

        #update alphas[j] and eCache[j]
        oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)

        if(abs(oS.alphas[j]-alphaJold)<0.00001):
           print("j not moving enought")
           return 0

        #update alphas[i] and eCache[i]
        oS.alphas[i]+=oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        updateEk(oS,j)

        #define b1 b2 to be choose
        #b1 substract Ei and index following labelMat[i] stay still but index following labelMat[j] changes
        # b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        ##use kernels for more complex data
        b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i]-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]

        # b2 substract Ej and index following labelMat[j] stay still but index following labelMat[i] changes
        # b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        ##use kernels for more complex data
        b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]

        #decide which b(b1 or b2) is to be use
        if(0<oS.alphas[i])and(oS.C>oS.alphas[i]):
            oS.b=b1
        elif(0<oS.alphas[j])and(oS.C>oS.alphas[j]):
            oS.b=b2
        else:
            oS.b=(b1+b2)/2.0
        return 1
    else:
        return 0

#outter loop
def smoP(dataMatIn,classLabels,C,toler,maxIter,KTup=('lin',0)):
    oS=optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,KTup)
    iter=0
    entireSet=True
    alphaPairsChanged=0
    while(iter<maxIter)and((alphaPairsChanged>0)or(entireSet)):
        alphaPairsChanged=0
        #decide how to use innerL()
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged+=innerL(i,oS)
            print("fullSet, iter: %d i: %d, paris changed %d"%(iter,i,alphaPairsChanged))
            iter+=1
        else:
            nonBoundIs=nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged+=innerL(i,oS)
                print("non-bound, iter: %d i: %d, paris changed %d"%(iter,i,alphaPairsChanged))
            iter+=1

        #decide how to modify entireSet
        if entireSet:
            entireSet=False
        elif(alphaPairsChanged==0):
            entireSet=True
        print("iteration number: %d"% iter)
    return oS.b,oS.alphas


dataArr,labelArr=loadDataSet('testSet.txt')
b,alphas=smoP(dataArr,labelArr,0.6,0.001,40)
# print(b)
# print(alphas[alphas>0])

#function to calculate ws( for the classification)
def calcWs(alphas,dataArr,classLabels):
    X=mat(dataArr)
    labelMat=mat(classLabels).transpose()
    m,n=shape(X)
    w=zeros((n,1))
    #the for loop go over the element in alphas,but only the support vector which are not zeros that matter
    for i in range(m):
        w+=multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

ws=calcWs(alphas,dataArr,labelArr)
print("ws:")
print(ws)

#classification
dataMat=mat(dataArr)
print(dataMat[0])#belongs to -1
print("class of dataMat[0]:"+str(dataMat[0]*ws+b))#-0.92555695<0 class=-1

print(dataMat[2])#belongs to 1
print("class of dataMat[2]:"+str(dataMat[2]*ws+b))# 2.62094609>0 class=1









































