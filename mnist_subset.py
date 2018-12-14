#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Author: Yi He, Saibal De
#Class: EECS545 Machine Learning
#Title: MNIST Data Load
#Date: 12-06-2018

import os.path
import mnist
import numpy as np

# Random seed
np.random.seed(0)

def init(digit1, digit2, num_train, num_test):
    # Download dataset
    if not os.path.isfile("mnist.pkl"):
        mnist.init()

    # Load whole dataset into memory
    x_train, t_train, x_test, t_test = mnist.load()

    # Subset training data
    if num_train > 0:
        indices1 = [i for i, j in enumerate(t_train) if ((j==digit1)or(j==digit2)) ]
        x = x_train[indices1,:]
        y = t_train[indices1]
        y = np.cast[int](y)
        y[y==digit1]=-1
        y[y==digit2]= 1
        ind1 = np.random.choice(np.arange(y.size), num_train)
        x = x[ind1,:]
        y = y[ind1]
    else:
        x = None
        y = None

    # Subset test data
    if num_test > 0:
        indices2 = [i for i, j in enumerate(t_test) if ((j==digit1)or(j==digit2)) ]
        xtest = x_test[indices2,:]
        ytest = t_test[indices2]
        ytest = np.cast[int](ytest)
        ytest[ytest==digit1]=-1
        ytest[ytest==digit2]=1
        ind2 = np.random.choice(np.arange(ytest.size),num_test)
        xtest = xtest[ind2,:]
        ytest = ytest[ind2]
    else:
        xtest = None
        ytest = None

    # Return
    return (x, y, xtest, ytest)
