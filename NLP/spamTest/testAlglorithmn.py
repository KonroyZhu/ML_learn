import numpy as np

# def read_file(path):
#     with open(path) as f:
#         word_vec=f.read()
#     return word_vec
#
# def get_trainin_set(word_vec):
#     data=[]
#     for vec in word_vec.split("\n"):
#         vec=vec.replace("[","").replace("]","")
#         vector=vec.split(",")
#         line=[]
#         for v in vector[:-1]:
#             line.append(int(v))
#         line.append(vector[-1].replace("'","").strip(" "))
#         # print(len(line))
#         data.append(line)
#     return data[:-1]
#
# def get_array(path):
#     word_vec=read_file(path)
#     _list=get_trainin_set(word_vec)[:-1]
#     _array=np.array(_list)
#     X=_array[:,:-1]
#     y=_array[:,-1]
#     return X,y
#
# FeatX,Featy=get_array("word_vec_train")
# print(FeatX)
# print(Featy)

def load_data_with_numpy(path):
    from sklearn import preprocessing
    dataset=np.loadtxt(path,delimiter=",")

    X=dataset[:,:-1]
    y=dataset[:,-1]
    return X,y


TrainX,Trainy=load_data_with_numpy("word_vec_train")
TestX,Testy=load_data_with_numpy("word_vec_test")



from sklearn.ensemble import ExtraTreesClassifier
def check_features_importance_by_TreeClassifier(X,y):
    model=ExtraTreesClassifier()
    model.fit(X,y)
    features_importance=model.feature_importances_
    return features_importance

features_importance=check_features_importance_by_TreeClassifier(TrainX,Trainy)
for importance in features_importance:
    print(importance)

def SVM():
    from sklearn.svm import SVC
    model=SVC()
    model.fit(TrainX,Trainy)
    preidicted=model.predict(TestX)
    expected=Testy

    hit=0
    for i in range(len(expected)):
        if expected[i]==preidicted[i]:
            hit+=1
    print(preidicted)
    print("SVM: "+str(hit/len(expected)))

SVM()
def NB():
    from sklearn.naive_bayes import GaussianNB
    model=GaussianNB()
    model.fit(TrainX,Trainy)
    preidicted=model.predict(TestX)
    expected=Testy

    hit=0
    for i in range(len(expected)):
        if expected[i]==preidicted[i]:
            hit+=1
    print(preidicted)
    print("navie bayes: "+str(hit/len(expected)))
NB()
def LR():
    from sklearn.linear_model import LogisticRegression
    model=LogisticRegression()
    model.fit(TrainX,Trainy)
    preidicted=model.predict(TestX)
    expected=Testy

    hit=0
    for i in range(len(expected)):
        if expected[i]==preidicted[i]:
            hit+=1
    print(preidicted)
    print("logistic regression: "+str(hit/len(expected)))
LR()






