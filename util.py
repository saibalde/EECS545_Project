#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

"""
subarray(array, row_indices, col_indices)

Create a subarray with given row and column indices

Note:
(1) This routine does not create a view
(2) The arguments must be NumPy arrays
"""
def subarray(array, row_indices, col_indices):
    assert type(array) == np.ndarray
    assert type(row_indices) == np.ndarray
    assert type(col_indices) == np.ndarray

    rows = [i for i in row_indices for j in col_indices]
    cols = [j for i in row_indices for j in col_indices]

    return array[(rows, cols)].reshape(row_indices.size, col_indices.size)
