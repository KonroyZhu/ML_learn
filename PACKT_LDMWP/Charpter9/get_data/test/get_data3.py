import requests
from lxml import etree

# url="http://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=0&rsv_idx=1&tn=baidu&wd=china&rsv_pq=e650102d00041aa3&rsv_t=78bcCPLSyVhbNsRk8Ek2imk2af1PAd4s90Ai0VGH0wIwy%2B0mwot2OVkrE3I&rqlang=cn&rsv_enter=1&rsv_sug3=6&rsv_sug1=5&rsv_sug7=100&rsv_sug2=0&inputT=2316&rsv_sug4=3159"
# r=requests.get(url)
# data=r.text
# # print(data)
#
# selector=etree.HTML(data)
# result=selector.xpath('//div[@class="c-abstract"]/text()')
# print(result)


url="http://www.gutenberg.org/files/1651/1651.txt"
r=requests.get(url)
data=r.text
print(data)
# path="/home/konroy/Documents/konroy/learning data mining with python/author-atrribute/burton/test"
# f=open(path,'w')
# f.write(data)



