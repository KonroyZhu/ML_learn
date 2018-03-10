#encoding=utf8
import os
from nltk.tokenize import WordPunctTokenizer


spam_list=os.listdir("./enron1/spam")
ham_list=os.listdir("./enron1/ham")

train_spam=spam_list[:40]
train_ham=ham_list[:40]
test_spam=spam_list[31:41]
test_ham=ham_list[31:41]

def get_text_list(path_suffix_list,name):
    train_text_list = []
    for suffix in path_suffix_list :
        with open("./enron1/"+name+"/"+suffix) as f:
            train_text_list.append(f.read())
    return train_text_list

stop_word=(open("./enron1/stop_word").read())
# print(stop_word)

train_ham_text_list=get_text_list(train_ham,"ham")
train_spam_text_list=get_text_list(train_spam,"spam")
test_ham_text_list=get_text_list(test_ham,"ham")
test_spam_text_list=get_text_list(test_spam,"spam")


def get_words_bag():
    words_bag=set([])
    train_list=[]
    for t_index in range(len(train_ham_text_list)):
        word_list=WordPunctTokenizer().tokenize(train_ham_text_list[t_index])
        train_list.append(word_list)
    for t_index in range(len(train_ham_text_list)):
        word_list=WordPunctTokenizer().tokenize(train_ham_text_list[t_index])
        train_list.append(word_list)
    # print(len(train_list))
    for doc in train_list:
        words_bag=words_bag|set(doc)
    return list(words_bag)

word_bag=get_words_bag()

def get_words_vec(text_list):
    word_vec=[]
    for w_index in range(len(word_bag)):
        amount=0
        word_list = WordPunctTokenizer().tokenize(text_list)
        for w in word_list:
            if word_bag[w_index]==w: amount+=1
        if word_bag[w_index] in word_list:
            word_vec.append(str(amount))
        else:
            word_vec.append(str(amount))
    return word_vec
def write_words_vec(file,ham_list,spam_list):
    for h_index in range(len(ham_list)):
        print("writeing file ham file "+str(h_index)+"'s vector ")
        file.write(','.join(get_words_vec(ham_list[h_index]))+",1"+"\n")
    for s_index in range(len(spam_list)):
        print("writeing file spam file " + str(s_index )+ "'s vector ")
        file.write(','.join(get_words_vec(spam_list[s_index]))+",0"+"\n")

with open("word_vec_train",'a') as f1:
    write_words_vec(f1,train_ham_text_list,train_spam_text_list)
with open("word_vec_test",'a') as f2:
    write_words_vec(f2,test_ham_text_list,test_spam_text_list)



# def get_word_bag():
#     word_bag = []
#     for mail in train_ham_text_list:
#         word_list=WordPunctTokenizer().tokenize(mail)
#         for word in word_list:
#             if word not in stop_word and word not in word_bag:
#                 word_bag.append(word)
#     for mail in train_spam_text_list:
#         word_list=WordPunctTokenizer().tokenize(mail)
#         for word in word_list:
#             if word not in stop_word and word not in word_bag:
#                 word_bag.append(word)
#     return word_bag
#
#
# word_bag=get_word_bag()
# # print(len(word_bag))
#
# def get_vec(word_text_list,word_bag_len,file,key):
#     if key=="ham": mark=1
#     else :mark=0
#     for index in range(len(word_text_list)):
#         word_vec=[]
#         for b_index in range(word_bag_len):
#             amount=0
#             word_word=WordPunctTokenizer().tokenize(train_spam_text_list[index])
#             for w_index in range(len(word_word)):
#                 if word_bag[b_index]==word_word[w_index]:
#                     amount+=1
#             word_vec.append(amount)
#         word_vec.append(mark)
#         print(len(word_vec))
#         file.write(str(word_vec).replace("[","").replace("]","").strip()+"\n")
#         print("geting word_vec "+str(index)+" from "+key+" list")
#
# def get_word_vec(file,ham_text_list,spam_text_list):
#     word_vec_list = []
#     word_bag_len = len(word_bag)
#     get_vec(ham_text_list,word_bag_len,file,"ham")
#     get_vec(spam_text_list,word_bag_len,file,"spam")







