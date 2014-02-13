#! /usr/bin/python

# usage: python topics.py <beta file> <vocab file> <num words>
#
# <beta file> is output from the lda-c code
# <vocab file> is a list of words, one per line
# <num words> is the number of words to print from each topic

import sys
import numpy as np
import re
def print_topics(beta_file, vocab_file, nwords, nUser, nProd):

    # get the vocabulary

    vocab = file(vocab_file, 'r').readlines()
    # vocab = map(lambda x: x.split()[0], vocab)
    vocab = map(lambda x: x.strip(), vocab)

    # for each line in the beta file
    output = open('topics.html','w')
    output.write('<html>\n')
    output.write('<style type="text/css"> .topic{	font-family:Courier New, monospace	; display: inline-block;	width: 200px;}</style>\n')
    
    indices = range(len(vocab))
    topic_no = 0
    for topic in file(beta_file, 'r'):
        output.write('<div class="topic">\n')

        head =  '<b>User %d to Prod %d</b>' % (topic_no/nUser, topic_no%nProd)
        print head
        output.write(head+'<br>\n')
        topic = map(float, topic.split())
        indices.sort(lambda x,y: -cmp(topic[x], topic[y]))
        for i in range(nwords):
            line = '{:8s} {:6f}'.format(vocab[indices[i]],(topic[indices[i]]))
            #line = '%s   %f' % (vocab[indices[i]],(topic[indices[i]]))
            line = re.sub(' ','&nbsp;',line)
            print line
            output.write(line+'<br>\n')
        output.write('<br><br>\n')
        output.write('</div>')
        topic_no = topic_no + 1
        print '\n'
        
    output.write('</html>\n')
    output.close()

if (__name__ == '__main__'):

    if (len(sys.argv) != 5):
       print 'usage: python topics.py <beta-file> <vocab-file> <num user> <num prod>\n'
       sys.exit(1)

    beta_file = sys.argv[1]
    vocab_file = sys.argv[2]
    nUser = int(sys.argv[3])
    nProd = int(sys.argv[4])
    nwords = 10
    print_topics(beta_file, vocab_file, nwords,nUser,nProd)
