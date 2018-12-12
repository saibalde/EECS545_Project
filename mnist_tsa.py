#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
import time
import matplotlib.pyplot as plt

from mnist_to_graph import initialize_graph
from lp import LP
from TSA import query_with_TSA

np.random.seed(0)

# Initialize Graph
num_train = 1000
num_test = 0
sigma = 2.0e3
graph, labels = initialize_graph(0, 1, num_train, num_test, sigma)

# Randomly initialize some labels
num_init_labels = 1
init_labels = np.random.choice(np.arange(0, num_train), num_init_labels,
                               replace=False)
for i in init_labels:
    graph.set_label(i, labels[i])

# Initial label propagation
LP(graph)

# Run the TSA algorithm
num_max_queries = 99
accuracy = np.zeros(num_max_queries, dtype=np.float)

for i in range(num_max_queries):
    # TSA step: find next node to query label
    t0 = time.time()
    queried_index = query_with_TSA(graph)
    tsa_time = time.time() - t0

    # query the oracle
    label = labels[queried_index]

    # update the graph with true label
    graph.set_label(queried_index, label)

    # LP step: predict labels
    t0 = time.time()    
    LP(graph)
    lp_time = time.time() - t0

    # compute training error and stop if done
    accuracy[i] = graph.accuracy(labels)

    # print some information
    print('Iteration: ', i + 1)
    print('    TSA | Time: ', tsa_time, '; Queried index: ', queried_index)
    print('     LP | Time: ', lp_time, '; Accuracy: ', accuracy[i])


num_queries = num_init_labels + np.arange(num_max_queries)
plt.plot(num_queries, accuracy)
plt.savefig('tsa_accuracy.pdf')
