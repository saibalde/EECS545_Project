#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from graph import Graph, subarray

def LP(graph, q = None):
    W = graph.weights
    l = graph.l
    u = graph.u

    d = np.sum(W, axis=1)
    DuuInv = 1.0 / d[u, np.newaxis]

    Puu = DuuInv * subarray(W, u, u)
    Pul = DuuInv * subarray(W, u, l)
    nLuuInv = np.linalg.inv(np.eye(len(u)) - Puu)

    graph.LuuInv = nLuuInv * DuuInv

    fu = np.matmul(nLuuInv, np.matmul(Pul, graph.labels[l]))

    if q == None:
        yu = np.zeros(fu.shape, dtype=np.int)
        yu[fu < 0] = -1
        yu[fu >= 0] = 1

        graph.labels[u] = yu
    else:
        fu_p = 1.0 + fu
        fu_m = 1.0 - fu

        inds = (q * fu_p / fu_p.sum()) > ((1.0 - q) * fu_m / fu_m.sum())

        yu = np.zeros(fu.shape, dtype=np.int)
        yu[inds] = 1
        yu[np.logical_not(inds)] = -1

        graph.labels[u] = yu
