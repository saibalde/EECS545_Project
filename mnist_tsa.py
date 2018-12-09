#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from graph import Graph
from tsa import TSA
from lp import LP

# Initialize Graph
graph = ...
labels = ...

# Randomly initialize some labels
...

# Run the TSA algorithm
while True:
    # predict labels
    LP(graph)

    # compute training error and stop if done
    training_error = ...
    
    if error < 1.0e-06:
        break

    # compute next node to label
    index = TSA(graph)

    # query the oracle
    label = labels[index]

    # update the graph with true label
    graph.set_label(index, label)

# predict on test set and compute error
...
