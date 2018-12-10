#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import mnist_to_graph

from graph import Graph
from lp import LP
from tsa import TSA

# Initialize Graph
graph, labels, num_train, num_test = mnist_to_graph.init(4, 9)

# Randomly initialize some labels
for i in range(25):
    index = random.randint(0, num_train - 1)
    label = labels[index]
    graph.set_label(index, label)

# Run the TSA algorithm
while True:
    # predict labels
    LP(graph)

    # compute training error and stop if done
    training_error = ...
    
    if error < 1.0e-06:
        break

    # compute next node to label
    queried_index = query_with_TSA(graph)

    # query the oracle
    label = labels[queried_index]

    # update the graph with true label
    graph.set_label(queried_index, label)

# predict on test set and compute error
...
