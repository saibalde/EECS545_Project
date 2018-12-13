#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 09:17:19 2018

@author: dzdang
"""

import numpy as np
from graph import Graph
from TSA import TSA,  query_with_TSA

# nNodes = 100
# N = np.arange(0,nNodes)

# stop = 50
# ell = N[0:stop]
# u = N[stop:nNodes]
# y_ell = np.zeros([len(ell),1])
# y_ell[:] = 1

# d = np.zeros((4,1),dtype=np.int8)
# d[0,:] = 5
# d[2,:] = 2
# l = [i for i in range(2)];
# d[l] = 10
# p = d[l];
# print(type(p))
# print(len(l))

# graph_al = Graph(0)
# query_with_TSA(graph_al)
np.random.seed(0)
a = np.random.randn(5,5)
norms = np.sum(a**2,axis=0)
print(a)
print(norms)
print(sum(a[:,0]**2))
