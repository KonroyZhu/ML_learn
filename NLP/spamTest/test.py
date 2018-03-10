import numpy as np

dataset=np.loadtxt("word_vec_train",delimiter=",")
X=dataset[:,:-1]
y=dataset[:,-1]

testset=np.loadtxt("word_vec_test",delimiter=",")
tX=testset[:,:-1]
ty=testset[:,-1]

from sklearn.decomposition import PCA

pca = PCA(n_components=4)#dimensionality reduce from 1074 to 4
X_r = pca.fit(X).transform(X)
# print(X_r)
# print(y)
tX_r=pca.fit(tX).transform(tX)
# print(tX_r)



from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_r,y)
predicted=model.predict(tX_r)
hit=0
expected=ty
for p_index in range(len(predicted)):
    if predicted[p_index]==expected[p_index]:
        hit+=1
print("Bayes with dimensionality reduction hit rate: "+str(hit/len(predicted)))

from sklearn.svm import SVC
model=SVC()
model.fit(X_r,y)
predicted=model.predict(tX_r)
hit=0
expected=ty
for p_index in range(len(predicted)):
    if predicted[p_index]==expected[p_index]:
        hit+=1
print("svm with dimensionality reduction hit rate: "+str(hit/len(predicted)))

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X,y)
predicted=model.predict(tX)
hit=0
expected=ty
for p_index in range(len(predicted)):
    if predicted[p_index]==expected[p_index]:
        hit+=1
print("Bayes without dimensionality reduction hit rate: "+str(hit/len(predicted)))

from sklearn.svm import SVC
model=SVC()
model.fit(X,y)
predicted=model.predict(tX)
hit=0
expected=ty
for p_index in range(len(predicted)):
    if predicted[p_index]==expected[p_index]:
        hit+=1
print("svm without dimensionality reduction hit rate: "+str(hit/len(predicted)))

