#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from graph import Graph

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

def LP(graph):
    W = graph.weights
    l = graph.l
    u = graph.u

    d = np.sum(W, axis=1)
    DuuInv = 1.0 / d[u, np.newaxis]

    Puu = DuuInv * subarray(W, u, u)
    Pul = DuuInv * subarray(W, u, l)
    nLuuInv = np.linalg.inv(np.eye(len(u)) - Puu)

    graph.LuuInv = nLuuInv * DuuInv
    graph.fu = - np.matmul(graph.LuuInv, np.matmul(Lul, graph.labels[l]))

    yu = np.zeros(fu.shape, dtype=np.int8)
    yu[fu < 0] = -1
    yu[fu >= 0] = 1

    graph.labels[u] = yu
