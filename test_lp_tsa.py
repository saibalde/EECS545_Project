#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Author: Saibal De
#Class: EECS545 Machine Learning
#Title: LP with TSA Queries
#Date: 12-12-2018

import numpy as np

import mnist_subset
import mnist_graph

from lp import LP
from TSA import query_with_TSA

x_train, y_train, _, _ = mnist_subset.init(4, 9, 2500, 0)
graph = mnist_graph.init(x_train)
y_train = (1 + y_train) / 2

inds = np.random.choice(np.arange(2500), 10, replace=False)
for i in inds:
    graph.set_label(i, y_train[i])

LP(graph)

accuracy = np.zeros(140, dtype=np.float)
num_query = 10 + np.arange(1, 141)
for t in range(140):
    index = query_with_TSA(graph)
    graph.set_label(index, y_train[index])
    LP(graph)
    accuracy[t] = graph.accuracy(y_train)

np.savez("test_lp_tsa.npz", num_query=num_query, accuracy=accuracy)
