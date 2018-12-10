#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module for realizing fully connected graphs

This module defines a class for constructing a fully connected graphs.
-   The edge weights are implemented using an adjacency matrix.
-   The class also stores the graph Laplacian.
"""

import numpy as np
import bisect

def subarray(array, row_indices, col_indices):
    """Create a subarray with given row and column indices

    Parameters
    ----------
    array: numpy.ndarray
        Two dimensional NumPy array
    row_indices: set
        Set of row indices to select
    col_indices: set
        Set of column indices to select

    Returns
    -------
    numpy.ndarray
        Two dimensional array constructed using selected rows and columns
    """
    rows = [i for i in row_indices for j in col_indices]
    cols = [j for i in row_indices for j in col_indices]
    return array[(rows, cols)].reshape(len(row_indices), len(col_indices))

"""
A class for encoding undirected, fully connected graphs
"""
class Graph:
    """Class realizing fully connected graphs.

    Attributes
    ----------
    num_nodes: int
        Number of nodes in the graph
    weights: numpy.ndarray
        Adjacency matrix with edge weights
    laplacian: numpy.ndarray
        Unnormalized graph Laplacian
    l: list
        Node indices for which the labels were queried from oracle so far
    u: list
        Node indices for which the labels were not queried so far
    labels: numpy.ndarray
        Labels of the nodes
    LuuInv: numpy.ndarray

    fu: numpy.ndarray

    """
    def __init__(self, num_nodes):
        """Construct a fully connected graph

        Create a graph with given number of nodes. Initially, the graph is
        fully disconnected (i.e. the edge weights are zero) and no predicted
        labels.

        Parameters
        ----------
        num_nodes: int
            Number of nodes on in the graph
        """
        self.num_nodes = num_nodes
        self.weights = np.zeros((num_nodes, num_nodes), dtype=np.float)
        self.laplacian = np.zeros((num_nodes, num_nodes), dtype=np.float)
        self.l = []
        self.u = [i for i in range(num_nodes)]
        self.labels = np.zeros(num_nodes, dtype=np.int8)
        self.LuuInv = None
        self.fu = None

    def set_weight(self, i, j, weight):
        """Set the weight of an edge
    
        Update the edge weight. This routine also updates the graph Laplacian
        accordingly.

        Parameters
        ----------
        i: int
            Node index
        j: int
            Node index
        weight: float
            Weight of the edge between the nodes
        """
        # retrive old weight
        #if i != j:
        #    old_weight = self.weights[i, j]

        # assign new edge weight
        self.weights[i, j] = weight
        if i != j:
            self.weights[j, i] = weight

        # update laplacian entries at (i, i), (i, j), (j, i) and (j, j)
        #if i != j:
        #    self.laplacian[i, i] += weight - old_weight
        #    self.laplacian[i, j] = -weight
        #    self.laplacian[j, i] = -weight
        #    self.laplacian[j, j] += weight - old_weight

    def compute_laplacian(self):
        d = np.sum(self.weights, axis=1)
        self.laplacian = -self.weights
        self.laplacian[np.diag_indices(self.num_nodes)] += d

    def set_label(self, i, label):
        """Label the specified node of the graph

        Parameters
        ----------
        i: int
            Node index
        label: int (+1, 0 or -1)
            Label of the node
        """
        self.labels[i] = label
        bisect.insort(self.l, i)
        self.u.remove(i)

    def laplacian_uu_inv(self):
        """Return inverse of unlabelled-unlabelled block of graph Laplacian

        Returns
        -------
        numpy.ndarray

        """
        return self.LuuInv

    def laplacian_ul(self):
        return subarray(self.laplacian, self.u, self.l)

    def predicted_labels(self):
        """Return predicted labels for the unlabelled nodes of the graph

        Returns
        -------
        numpy.ndarray

        """
        return self.fu
