#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Author: Saibal De
#Class: EECS545 Machine Learning
#Title: LP with Random Queries
#Date: 12-12-2018

import numpy as np

import mnist_subset
import mnist_graph

from lp import LP

x_train, y_train, _, _ = mnist_subset.init(4, 9, 2500, 0)
graph = mnist_graph.init(x_train)
y_train = (1 + y_train) / 2

inds = np.random.choice(np.arange(2500), 150, replace=False)

accuracy = np.zeros(150, dtype=np.float)
num_query = np.arange(1, 151)
for i in range(150):
    graph.set_label(i, y_train[i])
    LP(graph)
    accuracy[i] = graph.accuracy(y_train)

np.savez("test_lp_random.npz", num_query=num_query, accuracy=accuracy)
