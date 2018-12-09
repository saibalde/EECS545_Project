#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import numpy as np

import mnist

def _download_data():
    if not os.path.isfile("mnist.pkl"):
        mnist.init()

def _load_subset(digit1, digit2):
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
    raise NotImplementedError

def init(digit1, digit2, sigma):
    _download_data()

    x_train, y_train, x_test, y_test = _load_subset(digit1, digit2)

    num_train = y_train.shape[0]
    num_test = y_test.shape[0]

    graph = _generate_graph(x_train, x_test, sigma)

    labels = np.hstack((y_train, y_test))

    return (graph, labels, num_train, num_test)
