#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

"""
A class for encoding undirected, fully connected graphs, without self-loops
"""
class Graph:
    """
    Graph(nNodes)

    Create a graph with given number of nodes. Initially, the graph is fully
    disconnected.
    """
    def __init__(self, nNodes):
        self.nNodes = nNodes
        self.edgeWt = np.zeros(nNodes * (nNodes - 1) // 2, dtype=np.float64)
        self.lapMat = np.zeros((nNodes, nNodes), dtype=np.float64)

    """
    Graph.edgeLinearIndex(i, j)

    Map the index for edge between nodes i and j to the linear index for the
    edge weight array
    """
    def edgeLinearIndex(self, i, j):
        if i == j:
            raise ValueError("Graph does not have self loops")

        # ensure that i < j
        if i > j:
            i, j = j, i

        # return the linear index
        return i * self.nNodes + j - (i + 1) * (i + 2) // 2

    """
    Graph.getEdgeWeight(i, j)

    Return the weight of edge between nodes i and j. Nodes are zero-indexed.
    """
    def getEdgeWeight(self, i, j):
        return self.edgeWt[self.edgeLinearIndex(i, j)]

    """
    Graph.setEdgeWeight(i, j, weight)

    Set the weight of edge between nodes i and j. Nodes are zero-indexed
    """
    def setEdgeWeight(self, i, j, weight):
        # retrive old weight
        old_weight = self.getEdgeWeight(i, j)

        # assign new edge weight
        self.edgeWt[self.edgeLinearIndex(i, j)] = weight

        # update laplacian entries at (i, i), (i, j), (j, i) and (j, j)
        self.lapMat[i, i] += weight - old_weight
        self.lapMat[i, j] = -weight
        self.lapMat[j, i] = -weight
        self.lapMat[j, j] += weight - old_weight

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
