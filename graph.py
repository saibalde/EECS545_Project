#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

"""
A class for encoding undirected, fully connected graphs
"""
class Graph:
    """
    Graph(num_nodes)

    Create a graph with given number of nodes. Initially, the graph is fully
    disconnected.
    """
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.weights   = np.zeros((num_nodes, num_nodes, dtype=np.float)
        self.laplacian = np.zeros((num_nodes, num_nodes), dtype=np.float)

    """
    Graph.set_weight(i, j, weight)

    Set the weight of edge between nodes i and j. Nodes are zero-indexed
    """
    def set_weight(self, i, j, weight):
        # retrive old weight
        old_weight = self.weights[i, j]

        # assign new edge weight
        self.weights[i, j] = weight
        self.weights[j, i] = weight

        # update laplacian entries at (i, i), (i, j), (j, i) and (j, j)
        self.laplacian[i, i] += weight - old_weight
        self.laplacian[i, j] = -weight
        self.laplacian[j, i] = -weight
        self.laplacian[j, j] += weight - old_weight

"""
A class for storing graphs with (partially) labelled graphs. Labels are
{+1, -1}, unlabelled nodes are indicated with a label of 0.
"""
class LabelledGraph(Graph):
    """
    LabelledGraph(nNodes)

    Create a graph with given number of nodes. Initially, the graph is fully
    disconnected and all nodes are unlabelled.
    """
    def __init__(self, nNodes):
        Graph.__init__(self, nNodes)
        self.labels = np.zeros(nNodes, dtype=np.float64)
        self.labelled_indices = np.

    """
    LabelledGraph.getLabel(i)

    Return the label of node i. Nodes are zero-indexed.
    """
    def getLabel(self, i):
        return self.labels[i]

    """
    LabelledGraph.setLabel(i, label)

    Set the label of node i. Nodes are zero-indexed.
    """
    def setLabel(self, i, label):
        self.labels[i] = label
