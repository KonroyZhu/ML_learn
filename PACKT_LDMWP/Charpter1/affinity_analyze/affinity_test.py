import numpy as np
from collections import  defaultdict

matrix=np.loadtxt('/home/konroy/PycharmProjects/PACKT_LDMWP/Charpter1/affinity_analyze/affinity_data.txt')#global
valid_rules=defaultdict(int)#global
invalid_rules=defaultdict(int)#global
num_occurances=defaultdict(int)#global
features={0:"bread",1:"milk",2:"cheeze",3:"apples",4:"bananas"}
i=1

def finding_features_amount():
    number_of_features=0
    for i in matrix[0]:number_of_features+=1
    print("the amount of the features is: "+str(number_of_features))
    return number_of_features

number_of_features = finding_features_amount()#global

def finding_rules(premise):
    for sample in matrix:
        # print (sample[0])
        if sample[premise]==0:continue
        num_occurances[premise]+=1
        for conclusion in range(number_of_features-1):
            if conclusion==premise :continue
            if sample[conclusion]==1:
                valid_rules[(premise,conclusion)]+=1
            else:
                invalid_rules[(premise,conclusion)]+=1

def print_rules(premise,conclusion,support,confidence,features,i):
    print("Rule {0}: If a person buys {1} they will also buys {2}".format(i,features[premise],features[conclusion]))
    print("Support: {0}".format(support[(premise,conclusion)]))
    print("Confidence: {0:.3f}".format(confidence[(premise,conclusion)]))


if __name__ == "__main__":

    for premise in range(number_of_features-1):
        finding_rules(premise)
        support=valid_rules
        confidence=defaultdict(float)
        for premise,conclusion in valid_rules.keys():
            rule=(premise,conclusion)
            confidence[rule]=valid_rules[rule]/num_occurances[premise]

            print_rules(premise,conclusion,support,confidence,features,i)
            i += 1