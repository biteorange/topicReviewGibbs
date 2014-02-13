# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:17:26 2014

@author: tiansheng
"""
import sys
import json
import re

if __name__ == '__main__':
    filename = sys.argv[1]
    output = open(sys.argv[2],'w')
    data = []
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))

    #parse busId, userId, score, time, n, review
    for example in data:
        userId = example['user_id']
        prodId = example['business_id']
        score = example['stars']
        score = '%d'%(score,)
        time = example['date']
        time = re.sub('[^0-9]','',time)
        reviews = example['text']
        text = re.sub('[^a-zA-Z ]+', '', reviews)
        text = re.sub(' +',' ',text)
        text = text.lower()
        nWords = "%d"%(len(reviews.split()),)
        newline = " ".join([userId, prodId, score, time, nWords, text])
        
        output.write(newline+'\n')
