# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:11:18 2014

assuming the input files are produced at it 2000
compute the marginal distributions
@author: tiansheng
"""

import numpy as np
tw = np.loadtxt('tw_2000',float)
uit = np.loadtxt('uit_2000',float)
uit = uit.reshape((10,10,20))

u = np.loadtxt('u_2000',float)
i = np.loadtxt('i_2000',float)

#it = uit.sum(0)
it = uit[2,:,:]
ut = uit[:,1,:]
#ut = uit.sum(1)

nTopics = 20
nUserT = 10
nItemT = 10

iw = np.matrix(it) * np.matrix(tw)
uw = np.matrix(ut) * np.matrix(tw)

np.savetxt('item_topics',iw)
np.savetxt('user_topics',uw)


