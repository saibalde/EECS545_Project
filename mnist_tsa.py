#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
import time

from  mnist_to_graph import initialize_graph
from lp import LP
from TSA import query_with_TSA

# Initialize Graph
num_train = 1000
num_test = 0
sigma = 2.0e3
graph, labels = initialize_graph(4, 9, num_train, num_test, sigma)

# Randomly initialize some labels
init_num_labels = 25
for i in range(init_num_labels):
    graph.set_label(i, labels[i])

# Initial label propagation
LP(graph)
accuracy = (graph.labels == labels).sum() / labels.size
print(len(graph.l), ' ', accuracy)

print(len(graph.u))

# Run the TSA algorithm
num_max_labels = 75
for i in range(num_max_labels):
    # compute next node to label
    # queried_index = init_num_labels+i
    t0 = time.time()
    queried_index = query_with_TSA(graph)
    t1 = time.time()
    print('TSA comp. time for iteration',i, 'is:',t1-t0)

    # query the oracle
    label = labels[queried_index]

    # update the graph with true label
    graph.set_label(queried_index, label)

    # predict labels

    t0 = time.time()    
    LP(graph)
    t1 = time.time()
    print('LP comp. time for iteration',i, 'is:',t1-t0)

    # compute training error and stop if done
    accuracy = (graph.labels == labels).sum() / labels.size

    print(len(graph.l), ' ', accuracy)

# predict on test set and compute error
# ...
