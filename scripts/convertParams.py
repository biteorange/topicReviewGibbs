# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:11:18 2014

assuming the input files are produced at iter 2000
compute the marginal distributions
@author: tiansheng
"""

import numpy as np

nTopics = 20
nUserT = 5
nItemT = 5
nWords = 5000

path = '../foods_res/foods'
#data = np.loadtxt('../arts.data',int)
#user_count = np.bincount(data[:,0])
#u_ind = (.0 + user_count) / np.sum(user_count)
#item_count = np.bincount(data[:,1])
#i_ind = (.0 + item_count) / np.sum(item_count)

tw = np.loadtxt(path+'_tw_1500',float)
#item = np.loadtxt('../arts_res/arts_i_2000',float)
#user = np.loadtxt('../arts_res/arts_u_2000',float)
uit = np.loadtxt(path+'_uit_1500',float)
uit = uit.reshape((nUserT,nItemT,nTopics))

# not sure, substract background topics
#tw = np.abs(tw - np.mean(tw,0))
#tw = [p/sum(p) for p in tw]

#u_ind = np.matrix(u_ind) * np.matrix(user)
#a = np.zeros((nItemT, nTopics))
#for i in range(nUserT):
#    a += uit[i,:,:]*u_ind[0,i]
#
#iw = np.matrix(a) * np.matrix(tw)
#np.savetxt('item.topics',iw)
#
#i_ind = np.matrix(i_ind) * np.matrix(item)
#a = np.zeros((nUserT, nTopics))
#for i in range(nItemT):
#    a += uit[:,i,:] * i_ind[0,i]
#uw = np.matrix(a) * np.matrix(tw)
#np.savetxt('user.topics',uw)

uiw = np.matrix(uit.reshape(nUserT*nItemT,nTopics)) * np.matrix(tw)
np.savetxt('user_item.topics',uiw)
#for u in range(10):
#    ui = uiw[range(u,100,nItemT),:]
#    np.savetxt('fixed_prod_%d.topics'%(u,), ui)
#
#
#    
#ui_max_topics = np.argmax(uit, 2)
#print ui_max_topics




