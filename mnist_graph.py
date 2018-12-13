#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from graph import Graph

def init(x, sigma):
    n = x.shape[0]

    graph = Graph(n)

    for i in range(n):
        for j in range(i + 1, n):
            dij = np.linalg.norm(x[i, :] - x[j, :])
            wij = np.exp(-dij**2 / sigma**2)
            graph.set_weight(i, j, wij)
    
    graph.compute_laplacian()

    return graph
