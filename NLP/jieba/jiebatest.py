import jieba

# 功能 一）： 分词
print("功能 一）： 分词")
sentence1="奔利一看就是便宜货"
sentence2="我买了一把青菜"
sentence3="我是软工1503班的昆睿"
def segment(sen):
    for w in jieba.cut(sen):
        print(w)

segment(sentence1)
segment(sentence2)
segment(sentence3)

# 功能 2) ：添加自定义词典
# jieba.load_userdict() # file_name为自定义词典的路径

# 功能 3) ：关键词提取
# topK为返回几个TF/IDF权重最大的关键词，默认值为20
print("功能 3) ：关键词提取")
paragraph="为节能降耗，8月浙江杭州富阳地区水泥企业停窑检修10天，杭州、绍兴地区熟料供给趋紧，区域大厂库存降至四到五成。同时杭州大马水泥因窑磨运转不正常，上周内报价上涨40元/吨。与此同时杭绍地区主导企业南方通知上调低标袋散及高标袋装水泥出厂20元/吨，其余厂家陆续跟调。在高标散装水泥调整方面，主要企业尚在协商推进。总体来看，近期高温天气开始消退，省内需求略有改善回升，但也有部分企业为稳定销量不愿过快提价，预计在八月底到九月以后行情全面进入上行通道。此外，浙江环保督查对水泥及建筑行业影响较小，未对各地价格行情生产明显影响。"
import jieba.analyse
print("20%:")
for word in jieba.analyse.extract_tags(paragraph,int(len(paragraph)*0.2)):
    print(word),

# 功能 4) : 词性标注（这是重点）
print("功能 4) : 词性标注（这是重点）")
import jieba.posseg as pseg
words =pseg.cut(paragraph)
for w in words:
    print (w.word,w.flag) 