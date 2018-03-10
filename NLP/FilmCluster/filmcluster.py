
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction

# 载入 nltk 的英文停用词作为“stopwords”变量
stopwords = nltk.corpus.stopwords.words('english')
# 载入 nltk 的 SnowballStemmer 作为“stemmer”变量
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
titles=open("film-title").read().split("\n")
synopses=open("film-synopses").read().split("BREAKS HERE")


# 这里我定义了一个分词器（tokenizer）和词干分析器（stemmer），它们会输出给定文本词干化后的词集合

def tokenize_and_stem(text):
    # 首先分句，接着分词，而标点也会作为词例存在
    '''
    在NLP中，我们对一句话或一个文档分词之后，一般要进行词干化处理。
    词干化处理就是把一些名词的复数去掉，动词的不同时态去掉等等类似的处理。
    '''
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # 过滤所有不含字母的词例（例如：数字、纯标点）
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # 首先分句，接着分词，而标点也会作为词例存在
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # 过滤所有不含字母的词例（例如：数字、纯标点）
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


# 非常不 pythonic，一点也不！
# 扩充列表后变成了非常庞大的二维（flat）词汇表
totalvocab_stemmed = []#经过词干化
totalvocab_tokenized = []#仅仅经过分词
for i in synopses:
    allwords_stemmed = tokenize_and_stem(i)  # 对每个电影的剧情简介进行分词和词干化
    totalvocab_stemmed.extend(allwords_stemmed)  # 扩充“totalvocab_stemmed”列表

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)
print("仅仅经过分词"+str(len(totalvocab_tokenized)))
print("经过词干化"+str(len(totalvocab_stemmed)))

#vocab_frame from panda以词干化的英文单词原型作index 未被词干化的分词结果作data
print("panda data frame:")
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
print (vocab_frame)

# Tf-idf 与文本相似度
'''
max_df：这个给定特征可以应用在 tf-idf 矩阵中，用以描述单词在文档中的最高出现率。假设一个词（term）在 80% 的文档中都出现过了，那它也许（在剧情简介的语境里）只携带非常少信息。
min_df：可以是一个整数（例如5）。意味着单词必须在 5 个以上的文档中出现才会被纳入考虑。在这里我设置为 0.2；即单词至少在 20% 的文档中出现 。因为我发现如果我设置更小的 min_df，最终会得到基于姓名的聚类（clustering）——举个例子，好几部电影的简介剧情中老出现“Michael”或者“Tom”这些名字，然而它们却不携带什么真实意义。
ngram_range：这个参数将用来观察一元模型（unigrams），二元模型（ bigrams） 和三元模型（trigrams）。参考n元模型（n-grams）。

'''
print("Tf-idf 与文本相似度:")
from sklearn.feature_extraction.text import TfidfVectorizer

# 定义向量化参数
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                   min_df=0.2, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))

print("type of synopses:")
print(synopses)
tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)  # 向量化剧情简介文本

print(tfidf_matrix)
terms = tfidf_vectorizer.get_feature_names()
print(len(terms))


# 余弦相似度
'''
dist 变量被定义为 1 – 每个文档的余弦相似度。
余弦相似度用以和 tf-idf 相互参照评价。
可以评价全文（剧情简介）中文档与文档间的相似度。
被 1 减去是为了确保我稍后能在欧氏（euclidean）平面（二维平面）中绘制余弦距离。
'''
print("余弦相似度")
from sklearn.metrics.pairwise import cosine_similarity

dist = 1 - cosine_similarity(tfidf_matrix)

print(len(cosine_similarity(tfidf_matrix)))


# K-means 聚类
'''
利用 tf-idf 矩阵，你可以跑一长串聚类算法来更好地理解剧情简介集里的隐藏结构。
我首先用 k-means 算法。这个算法需要先设定聚类的数目（我设定为 5）
。每个观测对象（observation）都会被分配到一个聚类，这也叫做聚类分配（cluster assignment）。
这样做是为了使组内平方和最小。接下来，聚类过的对象通过计算来确定新的聚类质心（centroid）
。然后，对象将被重新分配到聚类，在下一次迭代操作中质心也会被重新计算，直到算法收敛。
'''
print("K-means 聚类")
from sklearn.cluster import KMeans

num_clusters = 5

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()
print("K-means result (for each synopses):")
print(clusters)

# 利用 joblib.dump pickle 模型（softmax_model），一旦算法收敛，重载模型并分配聚类标签（labels）。
from sklearn.externals import joblib

# 注释语句用来存储你的模型
# 因为我已经从 pickle 载入过模型了
# joblib.dump(km,  'doc_cluster.pkl')
print("储模型")
joblib.dump(km,  'doc_cluster.pkl')
