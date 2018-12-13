#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from graph import Graph

def init(x, sigma = None):
    n = x.shape[0]
    graph = Graph(n)
    D = np.zeros((n, n), dtype=np.float)

    # Plain
    #for i in range(n):
    #    for j in range(i + 1, n):
    #        dij = np.linalg.norm(x[i, :] - x[j, :])
    #        wij = np.exp(-dij**2 / sigma**2)
    #        graph.set_weight(i, j, wij)

    # kNN type
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = np.linalg.norm(x[i, :] - x[j, :])
            D[j, i] = D[i, j]
        inds = D[i, :].argsort()[1:11]
        sigma_i = D[i, inds].mean()
        graph.weights[i, inds] = np.exp(-D[i, inds]**2 / (3.0 * sigma_i**2))
    graph.weights = 0.5 * (graph.weights + graph.weights.T)

    graph.compute_laplacian()

    return graph
