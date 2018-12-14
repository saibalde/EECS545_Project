#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Author: Saibal De
#Class: EECS545 Machine Learning
#Title: Plot the Accuracy Graphs
#Date: 12-13-2018

import numpy as np
import matplotlib.pyplot as plt

rand = np.load("test_lp_random.npz")
tsa  = np.load("test_lp_tsa.npz")
vm   = np.load("test_lp_vm.npz")

svm  = np.load("active_SVM.npz")
svm_num_queries = np.arange(11, 151)

plt.figure()
plt.plot(rand['num_query'], rand['accuracy'], label='Randomized')
plt.plot(tsa['num_query'],  tsa['accuracy'],  label='TSA')
plt.plot(vm['num_query'],   vm['accuracy'],   label='VM')
plt.xlabel('Number of queries')
plt.ylabel('Training accuracy')
plt.legend()
plt.savefig('graph_accuracy.pdf')

plt.figure()
plt.plot(svm_num_queries, 1. - svm['pasi'], label='Randomized')
plt.plot(svm_num_queries, 1. - svm['simp'], label='Closest to Boundary')
plt.plot(svm_num_queries, 1. - svm['kmed'], label='k-Medoids')
plt.xlabel('Number of queries')
plt.ylabel('Training accuracy')
plt.legend()
plt.savefig('svm_accuracy.pdf')
