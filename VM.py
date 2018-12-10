'''
Author: Alexandros Georgakopoulos
Class: EECS545 Machine Learning
Title: VM Implementation
Date: 12-08-2018
'''

'''
--------------------------------------------------------------
A Variance Minimization Criterion to Active Learning in Graphs
--------------------------------------------------------------
Input is the Laplacian L of a graph and a number l in {1,...,n}.
The nodes have true labels y_i known by the Oracle.
We have a budget of l nodes to ask the Oracle to tell us their label.
Problem: Which nodes to ask for? Call their set L_cal.
The following algorithm produces L_cal using an inductive method
Output is L_cal, a subset of {1,2,...,n} with l elements.

After that, LP takes this set and uses it to predict the labels on the rest of the nodes.
--------------------------------------------------------------
'''

import numpy as np
from numpy import linalg

from mnist_to_graph import initialize_graph
from lp import LP

n = 100
l = 25
graph, labels = initialize_graph(4, 9, n, 0, 2.0e3)
L = graph.laplacian

#import the graph here, only the Laplacian L is needed
#n=50 #n=|V|, numbers of nodes of the graph
#l=10 #l=budget of nodes we want to learn their true label
#import random
#L_r=np.random.rand(n,n) #example
#L=(L_r + L_r.T)/2 #example

#eigendecomposition with decreasing eigenvalues
def sorteigen(L):
    eigenValues, eigenVectors = np.linalg.eig(L)
    idx = np.argsort(eigenValues)
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return (eigenValues, eigenVectors)

#using L we define the following quantities used throughout the algorithm
X=sorteigen(L)[1] #L=XSigmaX^T, is it X_TSigmaX?
Sigma=np.diag(sorteigen(L)[0]) #the diagonal entries must decrease and the nxn entry must be zero
M=Sigma+np.identity(n) #M=Sigma+I
M_inv=np.linalg.inv(M) #used in equations (12) and (20)
X_cal=[(j,X[j,:]) for j in range(n)] #X_cal is a list of (j, j-th row of X)
K=[] #list to be filled with l tuples (i,p), i nodes to be predicted, p= ith row of X
def f(x):
    return -(x+1)/x
B_hat_inv=np.zeros((n,n)) #used in equations (12) and (15)
B_hat_inv[:n-1,:n-1]=np.diag(f(sorteigen(L)[0][:n-1])) #puts -(ev+1)/ev on the diagonal from 1 to n-1
A_0=M_inv-np.identity(n) #M^{-1}-I, used in equations (12) and (16)

#Finding the first point to label, equation (12)
def C(p,B): #quantity in right hand side of equation (15), used in equation (12)
    v_hat=np.append(p[:n-1], 0)
    den=1+np.dot(np.dot(v_hat.T,B),v_hat)
    return B-(1.0/den)*np.dot(np.dot(np.dot(v_hat.T,B),v_hat.T),B)

ind,p_1=X_cal[0]
max=np.trace(np.dot(C(p_1,B_hat_inv),M_inv)) #equation (12)
for m,p in X_cal[1:]: #this loop solves the argmax problem of equation (12)
    tr=np.trace(np.dot(C(p,B_hat_inv),M_inv))
    if max > tr:
        max=tr
        ind,p_1=m,p
K.append((ind,p_1))

#Recursively finding the next l-1 points to label
def D(p,A): #quantity in right hand side of equation (18), used in equation (20) for j>1
    den=1+np.dot(np.dot(p.T,A),p)
    return A-(1.0/den)*np.dot(np.dot(np.dot(A,p),p.T),A)

for j in range(1,l): #this loop recursively finds all the nodes of L_cal after the first
    if j==1:
        A_j_inv=C(p_1,B_hat_inv) #for j=1 we use equation (15)
    else:
        A_j_inv=D(new_p,A_j_inv) #for j>1 we use equation (18)
    #the next three lines initialise the loop that solves the optimization problem of equation (20)
    index,new_p=X_cal[0]
    den_2=1+np.dot(np.dot(new_p.T,A_j_inv),new_p) #equation (20)
    min=(1.0/den_2)*np.dot(np.dot(np.dot(np.dot(new_p.T,A_j_inv),M_inv),A_j_inv),new_p) #equation (20)
    for m,p in X_cal[1:]:  #this loop solves the argmin problem of equation (20)
        if (m,p) not in K:
            den_2=1+np.dot(np.dot(p.T,A_j_inv),p) #equation (20)
            frac=(1.0/den_2)*np.dot(np.dot(np.dot(np.dot(p.T,A_j_inv),M_inv),A_j_inv),p) #equation (20)
            if min <= frac:
                min= frac
                index,new_p=m,p
    K.append((index,new_p))

L_cal=np.sort([K[j][0] for j in range(l)]) #the indices in K, K must have length l here

print("L_cal = ", L_cal)

for i in L_cal:
    graph.set_label(i, labels[i])

#for i in range(l):
#    graph.set_label(i, labels[i])

LP(graph)

train_error = (graph.labels == labels).sum() / labels.size
print("Training error using VOpt+LP = ", train_error)

#import code
#code.interact(local=locals())
