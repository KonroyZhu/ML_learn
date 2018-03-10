
import pandas as pd
import  os

#Load the data=======================================================================================================
data_folder="/home/konroy/Documents/konroy/learning data mining with python/ml-100k"
ratings_filename=os.path.join(data_folder,"u.data")
#the dataset's element is delimited by "\t" and doesn't have a header, so we have to set it by ourself
all_ratings=pd.read_csv(ratings_filename,delimiter="\t",header=None,names=["UserID","MovieID","Rating","Datetime"])
# print(all_ratings[:5])#have to re set the "Datetime"
all_ratings["Datetime"]=pd.to_datetime(all_ratings['Datetime'],unit='s')

#bring in new feature1:Favorable
all_ratings["Favorable"]=all_ratings["Rating"]>3

#choose part of the data from all_ratings serve as the training set
ratings=all_ratings[all_ratings['UserID'].isin(range(200))]#here we choose 200 user from the front




#Apriori algorithmn=====================================================================================================

#step1:---------------------------------------------------------------------------------------------

#element1: "favorable_reviews_by_users"
favorable_ratings=ratings[ratings["Favorable"]]
favorable_reviews_by_users=dict((k,frozenset(v.values))for k,v in favorable_ratings.groupby("UserID")["MovieID"])
#(ps: each user(k) is follow by a set of movies(v) that they've reviewed, and the rating of each movie is higher than 3)

#element2: "num_favorable_by_movie"
num_favorable_by_movie=ratings[["MovieID","Favorable"]].groupby("MovieID").sum()
# print(num_favorable_by_movie.sort("Favorable",ascending=False)[:5])#rank the movie by the fans amount
#(ps: the fans(they've given a rating witch is higher than 3) amount for each movie)

#step2,3:--------------------------------------------------------------------------------------------
###initialize the dictionary to store the data we need
frequent_itemsets={}
min_support=50
frequent_itemsets[1]=dict((frozenset((movie_id,)),row["Favorable"])
                                        for movie_id,row in num_favorable_by_movie.iterrows()
                                        if row["Favorable"]>min_support)
###
from collections import defaultdict
import sys
def find_frequent_itemsets(favorable_reviews_by_users,k_1_itemsets,min_support):
    counts=defaultdict(int)#####    element1    #######  element2  ###########
    for user ,reviews,in favorable_reviews_by_users.items():#traverse the users and their reviews
        for itemset in k_1_itemsets:## check the items under each user
            if itemset.issubset(reviews):#user has review the movie
                for other_reviewed_movie in reviews-itemset:#find out the movie that user has scored but wasn't included in the itenset
                    current_supperset=itemset | frozenset((other_reviewed_movie,))
                    counts[current_supperset]+=1##to ensure more than 50 user has the same situation that when they've chosen the item they will love the rest
    return dict([(item,frequency) for  item,frequency in counts.items() if frequency>=min_support])

#step4:------------------------------------------------------------------------------------------------
for k in range(2,20):#k is the length of the conclusion(that is to say: the conclusion may include k movies
    cur_frequent_itemsets=find_frequent_itemsets(favorable_reviews_by_users,frequent_itemsets[k-1],min_support)
    frequent_itemsets[k]=cur_frequent_itemsets
    if len(cur_frequent_itemsets)==0:
        print("Did not find any frequent itemsets of length {0}".format(k))
        sys.stdout.flush()
        break
    else:
        print("I found {0} frequent itemsets of length {1}".format(len(cur_frequent_itemsets),k))
        sys.stdout.flush()
        print(frequent_itemsets[k])
del frequent_itemsets[1]#to extract rules we have to find more than one item to build the relation, so delete the data only have 1 item
# print(frequent_itemsets)



#the affinity movies have been sored in the dict frequent_items
#Extract the rules======================================================================================================

candidate_rules=[]
for itemset_length,itemset_content in frequent_itemsets.items():#
    # print("itemset_length: {0} itemset_content: {1}".format(itemset_length,itemset_counts))
    for itemset in itemset_content.keys():#get the keys from the dictionary witch represent the movieID
        for conclusion in itemset:
            premise=itemset-set((conclusion,))#premise is at the front and should be isolate from conclusion
            candidate_rules.append((premise,conclusion))
print("candidate_rules: {0}".format(candidate_rules[:5]))


#Calculate the support and confidence===================================================================================
correct_counts=defaultdict(int)
incorrect_counts=defaultdict(int)

for user,reviews in favorable_reviews_by_users.items():
    for candidate_rule in  candidate_rules:
        premise,conclusion=candidate_rule
        #test weather the  rule is right
        # if premise.issubest(reviews):#AttributeError: 'frozenset' object has no attribute 'issubest'
        if premise.issubest(reviews):
            if conclusion in reviews:
                correct_counts[candidate_rule]+=1
            else:
                incorrect_counts[candidate_rule]+=1

rule_confidence ={candidate_rule: correct_counts[candidate_rule]/
                                  float(correct_counts[candidate_rule]+incorrect_counts[candidate_rule])
                                for candidate_rule in candidate_rules}

from operator import itemgetter
sorted_confidence =sorted(rule_confidence.items(),key=itemgetter(1),reverse=True)



#### Convert the MovieID into movie name:
data_folder="/home/konroy/Documents/konroy/learning data mining with python/ml-100k"
movie_name_filename=os.path.join(data_folder,"u.item")
movie_name_data=pd.read_csv(movie_name_filename,delimiter="|",header=None,encoding="mac-roman")

movie_name_data.columns=["MovieID","Title","Release Date","Video Release","IMDB","<UNK>","Action","Adventure","Animation",
                         "Children's","Comedy","Crime","Documentary",
                         "Drama","Fantasy","Film-Noir","Horror","Musical","Mestery","Romance","Sci-Fi","Thriller","Wat","Western"]
# print(movie_name_data[:5])

def get_movie_name(movie_id):
    title_object=movie_name_data[movie_name_data["MovieID"]==movie_id]["Title"]
    # print(title_object.values[0])
    title=title_object.values[0]
    return title

###Output
def output():
    for index in range(5):
        print("Rule {0}".format(index+1))
        (premise,conclusion)=sorted_confidence[index][0]
        premise_name=",".join(get_movie_name(idx)  for idx in premise)
        conclusion_name=get_movie_name(conclusion)
        print("Rule: if a person recommends {0} they will also recommend {1}".format(premise_name,conclusion_name))
        print("Train_Confidence: {0:.3f}".format(rule_confidence[premise,conclusion]))
        print("Test_Confidence: {0:.3f}".format(test_confidence[premise, conclusion]))


#Assessment=============================================================================================================
#To test the algorithmn we have to use a set of data witch is not the same as the training one, so the code is as follow
#hints: the operator '~' means to do the opposite things
test_dataset=all_ratings[~all_ratings['UserID'].isin(range(200))]#so the data is from the rest of all_ratings

test_favorable=test_dataset[test_dataset["Favorable"]]#the movie whose rating is higher than 3
test_favorable_by_users=dict((k,frozenset(v.values) )  for k,v in test_favorable.groupby("UserID")["MovieID"])

correct_counts=defaultdict(int)
incorrect_counts=defaultdict(int)
for user,reviews in test_favorable_by_users.items():
    for candidate_rule in candidate_rules:
        premise,conclusion=candidate_rule
        if premise.issubest(reviews):
            correct_counts[candidate_rule]+=1
        else:
            incorrect_counts[candidate_rule]+=1
test_confidence={candidate_rule: correct_counts[candidate_rule]/
                 float(correct_counts[candidate_rule]+incorrect_counts[candidate_rule])
                 for candidate_rule in rule_confidence}

output()




