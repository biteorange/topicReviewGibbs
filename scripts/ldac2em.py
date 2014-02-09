# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:18:41 2014

@author: tiansheng
"""
import sys

def writeToEM(filename):
    lines = open(filename,'r').readlines()
    file_out = filename.split('.')[0] + '.em'
    output = open(file_out,'w')
    doc = 0
    for line in lines:
        line = line.strip()
        words = line.split()
        # skip the unique word count
        for i in range(1,len(words)):
            wordCount = words[i].split(':')
            word = wordCount[0]
            count = int(wordCount[1])
            for k in xrange(count):
                output.write('%d'%(doc,)+' -1 '+word+'\n')
        doc = doc + 1
        
if __name__ == '__main__':
    filename = sys.argv[1]
    writeToEM(filename)
    
