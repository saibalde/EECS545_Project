#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import mnist
import numpy as np

# Random seed
np.random.seed(0)

def init():
    # Download dataset
    if not os.path.isfile("mnist.pkl"):
        mnist.init()

    # Load whole dataset into memory
    x_train, t_train, x_test, t_test = mnist.load()

    # Subset parameters
    digit1    = 4
    digit2    = 9
    num_train = 1000
    num_test  = 500

    # Subset training data
    indices1 = [i for i, j in enumerate(t_train) if ((j==digit1)or(j==digit2)) ]
    x = x_train[indices1,:]
    y = t_train[indices1]
    y = np.cast[int](y)
    y[y==digit1]=-1
    y[y==digit2]= 1
    ind1 = np.random.choice(np.arange(y.size), num_train)
    x = x[ind1,:]
    y = y[ind1]

    # Subset test data
    indices2 = [i for i, j in enumerate(t_test) if ((j==digit1)or(j==digit2)) ]
    xtest = x_test[indices2,:]
    ytest = t_test[indices2]
    ytest = np.cast[int](ytest)
    ytest[ytest==digit1]=-1
    ytest[ytest==digit2]=1
    ind2 = np.random.choice(np.arange(ytest.size),num_test)
    xtest = xtest[ind2,:]
    ytest = ytest[ind2]

    # Return
    return (x, y, xtest, ytest)
