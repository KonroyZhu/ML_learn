
import jieba

class ToSegSenList:
    def read_file(self,path):
        with open(path) as f:
            return f.read()


    def get_seg(self,sen_list,stopwords):
        seg_list=[[word for word in jieba.cut(document) if word not in stopwords]
                  for document in sen_list]
        return seg_list

    def get_seg_list(self,path):
        sen_list=self.read_file(path=path).split("\n")
        stopwords=self.read_file("./stop_words.txt")
        return self.get_seg(sen_list=sen_list,stopwords=stopwords)


c=ToSegSenList()
pos_list=c.get_seg_list("./内饰正.txt")
print(pos_list)