#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 09:17:19 2018

@author: dzdang
"""

import numpy as np
from graph import Graph
from TSA import TSA

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

def query_with_TSA(graph):
   TSA_instance = TSA(graph)
   TSA_instance.calc_lookahead_risk()
   q = TSA_instance.solve_eem()

   return q
   # q,y_q = TSA_instance.solve_eem()
