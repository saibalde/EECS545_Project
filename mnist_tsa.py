#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

from  mnist_to_graph import initialize_graph
from lp import LP
from tsa import TSA

# Initialize Graph
num_train = 100
num_test = 0
sigma = 2.0e3
graph, labels = initialize_graph(4, 9, num_train, num_test, sigma)

# Randomly initialize some labels
init_num_labels = 25
for i in range(init_num_labels):
    index = random.randint(0, num_train - 1)
    label = labels[index]
    graph.set_label(index, label)

# Initial label propagation
LP(graph)
accuracy = (graph.labels == labels).sum() / labels.size()
print(len(graph.l), ' ', accuracy)

# Run the TSA algorithm
num_max_labels = 75
for i in range(num_max_labels)
    # compute next node to label
    queried_index = query_with_TSA(graph)

    # query the oracle
    label = labels[queried_index]

    # update the graph with true label
    graph.set_label(queried_index, label)

    # predict labels
    LP(graph)

    # compute training error and stop if done
    accuracy = (graph.labels == labels).sum() / labels.size

    print(len(graph.l), ' ', accuracy)

# predict on test set and compute error
# ...
