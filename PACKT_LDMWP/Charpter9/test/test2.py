import requests
from lxml import etree

url="http://www.gutenberg.org/files/2349/2349.txt"
r=requests.get(url)
data=r.text
print(data)

selector=etree.HTML(data)
contnet=selector.xpath('//body/pre/text()')
