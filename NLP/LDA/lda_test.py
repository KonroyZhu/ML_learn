


####load stop words
stopwords=[]
with open("stopwords") as f:
    words=f.read()
for w in words:
    stopwords.append(w)
print("stopwords:\n")
print(stopwords)

####load text file
paragraph=""
with open("passage") as f:
    paragraph=f.read()
print("paragraph:\n"+paragraph)


####segment
import jieba
texts=[]
words_bag=[]
text=jieba.cut(paragraph)
for w in text:
    if w not in stopwords:
        words_bag.append(w)
texts.append(words_bag)



####lda
from gensim import corpora,models
import gensim
dictionary=corpora.Dictionary(texts)
print(dictionary)
corpus=[dictionary.doc2bow(t) for t in texts]
###generate LDA softmax_model
ldamodel=gensim.models.LdaModel(corpus,num_topics=1,id2word=dictionary,passes=20)
print(ldamodel.print_topic(0))