# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:11:18 2014

assuming the input files are produced at it 2000
compute the marginal distributions
@author: tiansheng
"""

import numpy as np

nTopics = 20
nUserT = 10
nItemT = 10
nWords = 5000
tw = np.loadtxt('arts_tw_1500',float)
uit = np.loadtxt('arts_uit_1500',float)
uit = uit.reshape((nUserT,nItemT,nTopics))

uiw = np.matrix(uit.reshape(nUserT*nItemT,nTopics)) * np.matrix(tw)
uiw = uiw[range(4,100,10),:]
print uiw.shape
np.savetxt('ui_topics',uiw)



# iw = np.matrix(it) * np.matrix(tw)
# uw = np.matrix(ut) * np.matrix(tw)

# np.savetxt('item_topics',iw)
# np.savetxt('user_topics',uw)


