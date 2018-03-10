from numpy import *
from os import listdir
def image2Vect(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()#not readlines()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

