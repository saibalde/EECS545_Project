#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import matplotlib.pyplot as plt

import mnist_subset
import mnist_graph

from lp import LP
from TSA import query_with_TSA

# Parameters
digit1    = 4
digit2    = 9
num_train = 1000
num_test  = 0

# Load data and generate graph
x_train, y_train, _, _ = mnist_subset.init(digit1, digit2, num_train,
                                           num_test)
y_train = (1 + y_train) / 2
graph = mnist_graph.init(x_train)
q = (y_train == 1).sum() / y_train.size

print(graph.weights.max())

# Randomly initialize some labels
np.random.seed(0)
num_init_labels = 25
init_labels = np.random.choice(np.arange(0, num_train), num_init_labels,
                               replace=False)
for i in init_labels:
    graph.set_label(i, y_train[i])

# Initial label propagation
LP(graph, q)

# Run the TSA algorithm
num_max_queries = 125
accuracy = np.zeros(num_max_queries, dtype=np.float)

t00 = time.time()
for i in range(num_max_queries):
    # TSA step: find next node to query label
    t0 = time.time()
    # queried_index = query_with_TSA(graph)
    queried_index = graph.u[len(graph.u)-1]
    tsa_time = time.time() - t0

    # query the oracle
    label = y_train[queried_index]

    # update the graph with true label
    graph.set_label(queried_index, label)

    # LP step: predict labels
    t0 = time.time()    
    LP(graph, q)
    lp_time = time.time() - t0

    # compute training error and stop if done
    accuracy[i] = graph.accuracy(y_train)

    # print some information
    print('Iteration: ', i + 1)
    print('    TSA | Time: ', tsa_time, '; Queried index: ', queried_index)
    print('     LP | Time: ', lp_time, '; Accuracy: ', accuracy[i])

np.savez("output.npz",accuracy=accuracy)

t11 = time.time()
print(t11-t00)

num_queries = num_init_labels + np.arange(num_max_queries)
plt.plot(num_queries, accuracy)
plt.savefig('tsa_accuracy.pdf')
