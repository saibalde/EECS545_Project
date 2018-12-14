#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Author: Saibal De
#Class: EECS545 Machine Learning
#Title: LP with VOpt Queries
#Date: 12-13-2018

import numpy as np

import mnist_subset
import mnist_graph

from lp import LP
from VM import VM

num_train = 2500
num_test  = 0
x_train, y_train, _, _ = mnist_subset.init(4, 9, num_train, num_test)
graph = mnist_graph.init(x_train)
y_train = (1 + y_train) / 2

num_query = np.arange(10, 151, 10)
accuracy = np.zeros(num_queries.size, dtype=np.float)

for j in range(num_queries.size):
    # Reset graph
    graph.u = [i for i in range(num_train)]
    graph.l = []

    # Run VM
    indices = VM(graph.laplacian, num_train, num_queries[j])

    # Query labels
    for i in indices:
        graph.set_label(i, y_train[i])

    # Run LP
    LP(graph)

    # Compute accuracy
    accuracy[j] = graph.accuracy(y_train)

np.savez("test_lp_vm.npz", num_query=num_query, accuracy=accuracy)
