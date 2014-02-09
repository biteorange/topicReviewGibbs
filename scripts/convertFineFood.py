# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 22:15:19 2014
command: python convertFineFood.py
note: check file exists
@author: tiansheng
"""
import re
# parse the finefood data
output = open("foods.data",'w')
i = 0
count = 0
newline = ""
lines = open('foods.txt','r').readlines()
nLines = len(lines)
count = 0
for i in range(0,nLines,9):
    count = count + 1
    if (count % 10000 == 0):# or count > 530000):
        print "processed %d files"%(count,)
        print lines[i+7]
    #if (count > 90000):
        #print lines[i+7]
    # write o
    if (i+9 >= nLines):
        break
    
    prodId = lines[i].split("productId: ")[1].replace('\n','')
    userId = lines[i+1].split("userId: ")[1].replace('\n','')
    score = lines[i+4].split("score: ")[1].replace('\n','')
    time = lines[i+5].split("time: ")[1].replace('\n','')
    reviews = lines[i+7].split("text: ")[1].replace('\n','')
    text = re.sub('[^a-zA-Z ]+', '', reviews)
    text = text.lower()
    nWords = "%d"%(len(reviews.split()),)
    newline = " ".join([userId, prodId, score, time, nWords, text])
    
    output.write(newline+'\n')
