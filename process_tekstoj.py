
# pip3 install lxml

import os
from lxml import etree

directory = './tekstoj'
originalArticles = os.listdir(directory)

count = 0
total = len(originalArticles)

for article in originalArticles:
    count = count + 1

    if (article.find('.xml') == -1):
        continue
    print(article)

    xmlsource = open(directory + '/' + article, 'r')
    htmltree = etree.HTML(xmlsource.read())
    content = htmltree.findall(".//p")
    xmlsource.close()

    if (len(content) > 0):
        txtsource = open(directory + '/' + article.replace('.xml', '') + '.txt', 'w')
        for para in content:
            if para.text is not None:
                txtsource.write(para.text + "\n\n")
        txtsource.close()
