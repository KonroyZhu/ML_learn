import jieba


pospath="./内饰正.txt"
negpath="./内饰负.txt"
def read_file(path):
    content=""
    with open(path) as f:
        content=f.read()
    return content
pos_list=read_file(pospath).split("\n")
neg_list=read_file(negpath).split("\n")
stop_words=read_file("./stop_words.txt").split("\n")



def get_segment(comm_list,stopword):
    texts = [[word for word in jieba.cut(document) if word not in stopword]
          for document in comm_list]
    return texts
print("#### 第一步，把待转换成词袋的词变成需要的类型")
pos_segment_list=get_segment(pos_list,stop_words)#将分好词的评论语句放入列表
neg_segment_list=get_segment(neg_list,stop_words)#将分好词的评论语句放入列表

from gensim import corpora
# 通过学习pos_segment_list + neg_segment_list得到我们的字典
#将正负评论分好词的句子列表作为输入
dictionary=corpora.Dictionary(pos_segment_list + neg_segment_list)
#借助gensim包可以轻松完成词语向量构建过程
print("#####第二步：把所有文档根据字典转换成VSM")
corpus=[dictionary.doc2bow(segment) for segment in pos_segment_list+neg_segment_list ]
# 现在得到的corpus利用了字典，把每一个文档变成了一个一个tuple组合的形式，key为ID，value为出现的频数
print(corpus[:3])
print("##### 第三步，把频数变为tfidf值【用corpus训练】")
from gensim import  models
tfdif=models.TfidfModel(corpus)

# 从正向评论中抽取了一句
a=pos_segment_list[6]
print(a)
print(dictionary.doc2bow(a))
print(tfdif[dictionary.doc2bow(a)])
print()



print("#### 第四步：用训练好的tfidf把测试文档变为tfidf格式")
corpus_tfdif=tfdif[corpus]
for vec in corpus_tfdif:
    print(vec)


print("#### 第五步：把规整后的tfidf矩阵放进LDA模型中")

lda=models.LdaModel(corpus=corpus_tfdif,id2word=dictionary,num_topics=59)
for topic in lda.print_topics(5):
    print(topic)
# print(dictionary)
# # for vec in corpus:
# #     print(vec)
#
# from gensim import models
# tfidf=models.TfidfModel(pos_corpus+neg_cropus)
# print(tfidf)
# print(tfidf[pos_corpus[0]])


def test():
    print("test")


        # def get_words_bag(path):
#     sentence_list=read_file(path).split("\n")
#     words_bag = set([])# set 可以去重
#     for sentence in sentence_list:
#         for word in jieba.cut(sentence):
#             if word not in read_file("./stop_words.txt").split("\n"):
#                 words_bag.add(word)
#     return words_bag
#
# # print(len(get_words_bag(pospath)))
# # print(len(get_words_bag(negpath)))
# # print(len(get_words_bag(pospath)|get_words_bag(negpath)))
# # words_bag=get_words_bag(pospath)|get_words_bag(negpath)