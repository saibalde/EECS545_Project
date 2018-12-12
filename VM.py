'''
Author: Alexandros Georgakopoulos
Class: EECS 545 Machine Learning
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

n = 300
l = 30
graph, labels = initialize_graph(4, 9, n, 0, 2.0e3)
L = graph.laplacian

#import the graph here, only the Laplacian L is needed
#n=50 #n=|V|, numbers of nodes of the graph
#l=10 #l=budget of nodes we want to learn their true label
# import random
# L_r=np.random.rand(n,n) #example
# L=(L_r + L_r.T)/2 #example

#eigendecomposition with decreasing eigenvalues
def sorteigen(L):
    eigenValues, eigenVectors = np.linalg.eig(L)
    idx = eigenValues.argsort()[::-1][:]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return (eigenValues, eigenVectors)

#using L we define the following quantities used throughout the algorithm
X=sorteigen(L)[1] #L=XSigmaX^T
Sigma=np.diag(sorteigen(L)[0]) #the diagonal entries must decrease and the nxn entry must be zero
M=Sigma+np.identity(n) #M=Sigma+I
M_inv=np.linalg.inv(M) #used in equations (12) and (20)
K=[]
def f(x):
    return -(x+1.0)/x
B_hat_inv=np.zeros((n,n)) #used in equations (12) and (15)
B_hat_inv[:n-1,:n-1]=np.diag(f(sorteigen(L)[0][:n-1])) #puts -(eigenvalue+1)/eigenvalue on the diagonal from 1 to n-1
A_0=M_inv-np.identity(n) #M^{-1}-I, used in equations (12) and (16)

#four auxiliary functions
def C(p,B): #quantity in right hand side of equation (15), used in equation (12)
    v_hat=np.append(p[:n-1], 0)
    denominator=1.0+np.dot(np.dot(v_hat.T,B),v_hat)
    return B-(1.0/denominator)*np.dot(np.dot(np.dot(B,v_hat),v_hat.T),B)
def D(p,A): #quantity in right hand side of equation (18), used in equation (20) for j>1
    denominator=1.0+np.dot(np.dot(p.T,A),p)
    return A-(1.0/denominator)*np.dot(np.dot(np.dot(A,p),p.T),A)
def quantity_20(A,t):
    return (1.0/1.0+np.dot(np.dot(X[t,:].T,A),X[t,:]))*np.dot(np.dot(np.dot(np.dot(X[t,:].T,A),M_inv),A),X[t,:])
def argmin_outside_K(A,K):
    min_index=list(set(range(n)) - set(K))[0] #initialisation for the loop below
    min_quantity=quantity_20(A,min_index)
    for t in range(n): #here using a loop for finding argmin
        if t not in K and min_quantity < quantity_20(A,t):
            min_index=t
            min_quantity=quantity_20(A,t)
    return min_index

#Finding the first point to label, equation (12)
#this solves the argmax problem of equation (12)
ind=np.argmax([np.trace(np.dot(C(X[j,:],B_hat_inv),M_inv)) for j in range(n)])
p_1=X[ind,:]
K.append(ind)

#Recursively finding the next l-1 points of L_cal to label
for j in range(1,l):
    if j==1:
        A_j_inv=C(p_1,B_hat_inv) #for j=1 we use equation (15)
    else:
        A_j_inv=D(new_p,A_j_inv) #for j>1 we use equation (18)
    index=argmin_outside_K(A_j_inv,K)
    new_p=X[index,:]
    K.append(index)

L_cal=K
print(L_cal)

# import code
# code.interact(local=locals())

for i in L_cal:
   graph.set_label(i, labels[i])

LP(graph)

train_error = (graph.labels == labels).sum() / labels.size
print("Training error using VOpt+LP = ", train_error)
