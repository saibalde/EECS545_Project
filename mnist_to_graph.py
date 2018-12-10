#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import numpy as np

import mnist

from graph import Graph

def _download_data():
    if not os.path.isfile("mnist.pkl"):
        mnist.init()

def load_subset(digit1, digit2):
    _download_data()

    x_train, y_train, x_test, y_test = mnist.load()

    train_subset = np.logical_or(y_train == digit1, y_train == digit2)

    x_train_subset = x_train[train_subset]
    y_train_subset = np.array(y_train[train_subset], dtype=np.int8)

    y_train_subset[y_train_subset == digit1] = -1
    y_train_subset[y_train_subset == digit2] = 1

    test_subset = np.logical_or(y_test == digit1, y_test == digit2)

    x_test_subset = x_test[test_subset]
    y_test_subset = np.array(y_test[test_subset], dtype=np.int8)

    y_test_subset[y_test_subset == digit1] = -1
    y_test_subset[y_test_subset == digit2] = 1

    return (x_train_subset, y_train_subset, x_test_subset, y_test_subset)

def _generate_graph(x_train, x_test, sigma):
    x = np.vstack((x_train, x_test))
    n = x.shape[0]
    graph = Graph(n)

    for i in range(n):
        for j in range(i + 1, n):
            dij = np.linalg.norm(x[i, :] - x[j, :])
            wij = np.exp(-dij**2 / sigma**2)
            graph.set_weight(i, j, wij)

    graph.compute_laplacian()

def init(digit1, digit2, sigma):
    x_train, y_train, x_test, y_test = load_subset(digit1, digit2)

    num_train = y_train.shape[0]
    num_test = y_test.shape[0]

    graph = _generate_graph(x_train, x_test, sigma)

    labels = np.hstack((y_train, y_test))

    return (graph, labels, num_train, num_test)