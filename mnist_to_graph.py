#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import numpy as np
import time

import mnist

from graph import Graph

def _download_data():
    if not os.path.isfile("mnist.pkl"):
        mnist.init()

def load_subset(digit1, digit2, num_train, num_test):
    _download_data()

    np.random.seed(0)

    x_train, y_train, x_test, y_test = mnist.load()

    # subset training dataset
    total_train = x_train.shape[0]
    train_subset = [i for i in range(total_train)
                    if y_train[i] == digit1 or y_train[i] == digit2]

    if len(train_subset) < num_train:
        raise RuntimeError
    train_subset = np.random.choice(train_subset, size=num_train,
                                    replace=False)
    #train_subset = tain_subset[0:num_train]

    x_train_subset = x_train[train_subset]
    y_train_subset = np.array(y_train[train_subset], dtype=np.int8)

    y_train_subset[y_train_subset == digit1] = -1
    y_train_subset[y_train_subset == digit2] = 1

    # subset test dataset
    total_test = x_test.shape[0]
    test_subset = [i for i in range(total_test)
                   if y_test[i] == digit1 or y_train[i] == digit2]

    if len(test_subset) < num_test:
        raise RuntimeError
    test_subset = np.random.choice(test_subset, size=num_test,
                                   replace=False)
    #test_subset = test_subset[0:num_test]

    x_test_subset = x_test[test_subset]
    y_test_subset = np.array(y_test[test_subset], dtype=np.int8)

    y_test_subset[y_test_subset == digit1] = -1
    y_test_subset[y_test_subset == digit2] = 1

    return (x_train_subset, y_train_subset, x_test_subset, y_test_subset)

def _generate_graph(x_train, x_test, sigma):
    x = np.vstack((x_train, x_test))
    n = x.shape[0]
    graph = Graph(n)

    x_norm = np.linalg.norm(x, axis=1)
    x = x / x_norm[:, np.newaxis]

    for i in range(n):
        for j in range(i + 1, n):
            dij = np.linalg.norm(x[i, :] - x[j, :])
            wij = np.exp(-dij**2 / sigma**2)

            graph.set_weight(i, j, wij)

    graph.compute_laplacian()

    return graph

def initialize_graph(digit1, digit2, num_train, num_test, sigma):
    x_train, y_train, x_test, y_test = load_subset(digit1, digit2, num_train,
                                                   num_test)

    graph = _generate_graph(x_train, x_test, sigma)

    labels = np.hstack((y_train, y_test))

    return (graph, labels)
