#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 09:17:19 2018

@author: dzdang
"""

import numpy as np
from graph import Graph
from TSA import TSA

nNodes = 100
N = np.arange(0,nNodes)

stop = 50
ell = N[0:stop]
u = N[stop:nNodes]
y_ell = np.zeros([len(ell),1])
y_ell[:] = 1

graph_instance = Graph(nNodes)

TSA_instance = TSA(ell,u,y_ell,graph_instance)

TSA_instance.calc_lookahead_risk()

TSA_instance.solve_eem