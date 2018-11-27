#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Some common utities to make subarray construction easier

Example
-------
The following example shows how to construct a mixed (labelled-unlabelled)
Laplacian submatrix:
    >>> labels = [0, 1, 0, 0, -1, 0, 0, -1, 1, 0]   # labels of 10 nodes
    >>> laplacian = np.random.randn(10, 10)         # laplacian matrix
    >>> l = labelled_nodes(l)
    >>> u = unlabelled_nodes(l)
    >>> laplacian_lu = subarray(laplacian, l, u)
"""

import numpy as np

def subarray(array, row_indices, col_indices):
    """
    Create a subarray with given row and column indices

    Parameters
    ----------
    array: np.ndarray
        Two dimensional NumPy array
    row_indices: np.ndarray
        One dimensional NumPy array of row indices to select
    col_indices: np.ndarray
        One dimensional NumPy array of column indices to select

    Returns
    -------
    np.ndarray
        A two dimensional subarray constructed using selected rows and
        columns
    """
    assert type(array) == np.ndarray
    assert type(row_indices) == np.ndarray
    assert type(col_indices) == np.ndarray

    rows = [i for i in row_indices for j in col_indices]
    cols = [j for i in row_indices for j in col_indices]

    return array[(rows, cols)].reshape(row_indices.size, col_indices.size)

def labelled_nodes(labels):
    """
    From a set of labels, find the node indices that are labelled

    Parameters
    ----------
    labels: np.ndarray
        One dimensional NumPy array of labels. It is assumed that non-zero
        means true label, zero means unlabelled

    Returns
    -------
    np.ndarray
        One dimensional NumPy array of indices of labelled nodes
    """
    return np.where(labels != 0)[0]

def unlabelled_nodes(labels):
    """
    From a set of labels, find the node indices that are unlabelled

    Parameters
    ----------
    labels: np.ndarray
        One dimensional NumPy array of labels. It is assumed that non-zero
        means true label, zero means unlabelled

    Returns
    -------
    np.ndarray
        One dimensional NumPy array of indices of unlabelled nodes
    """
    return np.where(labels == 0)[0]
