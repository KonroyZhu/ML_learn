from numpy import*
randArray=random.rand(4,4)


#convert array to a matrix
randMat=mat(randArray)

#.I inverse matrix---only matrix can use method .I( AB=BA=E)
invRandMat=randMat.I

print(randArray)
print(randMat)
print(invRandMat)
print(randMat*invRandMat)

