# load model
print("loading softmax_model")
from sklearn.externals import joblib
km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()



import pandas as pd
'''
#下面，我创建了一个字典，包含片名，简要剧情，聚类分配，还有电影类型（genre）
#（排名和类型是从 IMDB 上爬下来的）。
下面，我创建了一个字典，包含片名，简要剧情，聚类分配，
'''
titles=open("film-title").read().split("\n")
synopses=open("film-synopses").read().split("BREAKS HERE")

print(len(titles))
print(len(synopses))
print(clusters)

films = {'title': titles,  'synopsis': synopses, 'cluster': clusters}

frame = pd.DataFrame(films, index=[clusters], columns=[ 'title', 'cluster'])
print(frame)




'''
在这选取 n（我选 6 个） 个离聚类质心最近的词对聚类进行一些好玩的索引（indexing）和排列（sorting）。
这样可以更直观观察聚类的主要主题。
'''
##################################################################################
import re
import nltk



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
print("Tf-idf 与文本相似度:")
from sklearn.feature_extraction.text import TfidfVectorizer

# 定义向量化参数
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                   min_df=0.2, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))

tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)  # 向量化剧情简介文本

print(tfidf_matrix)
terms = tfidf_vectorizer.get_feature_names()


##################################################################################

num_clusters=5
print("Top 6 terms per cluster:")
print()
# 按离质心的距离排列聚类中心，由近到远
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
print(order_centroids)
for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')

    for ind in order_centroids[i, :6]:  # 每个聚类选 6 个词
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print()  # 空行
    print()  # 空行

    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print()  # 空行
    print()  # 空行