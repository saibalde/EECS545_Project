"""
Author: Yi He
Class: EECS545 Machine Learning
Title: Active SVM Implementation
Date: 12-09-2018
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm

'''
num_train = 2000
num_test = 100
from mnist_to_graph import *
x,y,xtest,ytest = load_subset(4,9,num_train, num_test)
######Finished intialize of data, but training time is too long 
'''

#import mnist
#mnist.init()
#x_train, t_train, x_test, t_test = mnist.load()

#indices1 = [i for i, j in enumerate(t_train) if ((j==4)or(j==9)) ]
#indices2 = [i for i, j in enumerate(t_test) if ((j==4)or(j==9)) ]
#x = x_train[indices1,:]
#y = t_train[indices1]
#y = np.cast[int](y)
#y[y==4]=-1
#y[y==9]= 1
#ind1 = np.random.choice(np.arange(y.size), 1000)
#x = x[ind1,:]
#y = y[ind1]
#
#xtest = x_test[indices2,:]
#ytest = t_test[indices2]
#ytest = np.cast[int](ytest)
#ytest[ytest==4]=-1
#ytest[ytest==9]=1
#ind2 = np.random.choice(np.arange(ytest.size),500)
#xtest = xtest[ind2,:]
#ytest = ytest[ind2]

import mnist_subset
x, y, xtest, ytest = mnist_subset.init()

'''
#### Useing data from class here
mnist_49_3000 = sio.loadmat('mnist_49_3000.mat')
X = mnist_49_3000['x']
Y = mnist_49_3000['y']
X = X.T
Y = Y[0,:]

numofx =1000
x = X[:numofx,:]    # X= x+xtest
y = Y[:numofx]
xtest = X[numofx:, :]
ytest = Y[numofx:]
'''

numoftrain = 10
##### futher divide x into xtrain & xact
xtrain = x[:numoftrain, :]
ytrain = y[:numoftrain]
xact = x[numoftrain:, :]
yact = y[numoftrain:]


#### Report true score (1-error) of xtrain & x
clf1 = svm.SVC(kernel='linear',C=1)
clf1.fit(xtrain, ytrain)
scoreOfxtrain = clf1.score(xtest,ytest)
print (scoreOfxtrain)

clf2 = svm.SVC(kernel='linear',C=1)
clf2.fit(x,y)
scoreOfx = clf2.score(xtest,ytest)
print(scoreOfx)


'''
ypred = clf.predict(xtest)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(ytest, ypred))  
print(classification_report(ytest, ypred))  

b = clf.support_vectors_
sample = xact[np.argmin(np.abs(clf.decision_function(xact)))] 
'''

###### To find out #of samples in xact within margin
margin = 1 / np.sqrt(np.sum(clf1.coef_ ** 2))
dist_sort = np.sort(np.abs(clf1.decision_function(xact)))
cut = np.searchsorted(dist_sort, margin) 
if cut >= len(xact)/4:
    cut = len(xact)/4
    cut = np.cast[int](cut)
    
idx_sort = np.argsort(np.abs(clf1.decision_function(xact)))
xact_sort = xact[idx_sort, : ]
yact_sort = yact[idx_sort]

n = 30

def passive_learning (xtrain, ytrain, xact, yact,n):
    print('passive')
    result = np.zeros(n)
    for i in range (1,n):
        xtrain_new = np.concatenate((xtrain, xact[:i,:]), axis=0)
        ytrain_new = np.concatenate((ytrain, yact[:i]), axis=0)      
        passi_learn = svm.SVC(kernel='linear',C=1)
        passi_learn.fit(xtrain_new, ytrain_new)
        score_passi = passi_learn.score(xtest,ytest)
        result[i] = score_passi
        print (score_passi)
        if (score_passi >= scoreOfx):
            print (i)
            break
    return(result)
    
def simple_margin(xtrain, ytrain, xact_sort, yact_sort,n):
    print('simple')
    result = np.zeros(n)
    for i in range (1,n) :
        xtrain_new = np.concatenate((xtrain, xact_sort[:i,:]), axis=0)
        ytrain_new = np.concatenate((ytrain, yact_sort[:i]), axis=0)
        act_learn = svm.SVC(kernel='linear',C=1)
        act_learn.fit(xtrain_new, ytrain_new)
        score_simple = act_learn.score(xtest,ytest)
        result[i] = score_simple
        print (score_simple)
        if (score_simple >= scoreOfx):
            print (i)
            break
    return (result)
       
def kmedoids_active_learning(xtrain,ytrain,xact_sort,yact_sort,cut,n):
    from sklearn.metrics.pairwise import pairwise_distances
    import kmedoids
    print('kmedoids')  
    result = np.zeros(n)
    for i in range (2,n):
        xact_sort = xact_sort[:cut,:]
        yact_sort = yact_sort[:cut]
        D = pairwise_distances(xact_sort, metric='euclidean')
        M, C = kmedoids.kMedoids(D, i)
        xact_medoids = xact_sort[M, :]
        yact_medoids = yact_sort[M]
        xtrain_new = np.concatenate((xtrain, xact_medoids), axis=0)
        ytrain_new = np.concatenate((ytrain, yact_medoids), axis=0)
        act_learn = svm.SVC(kernel='linear',C=1)
        act_learn.fit(xtrain_new, ytrain_new)
        score_km = act_learn.score(xtest,ytest)
        result[i] = score_km
        print(score_km)
    return (result)

pasi = passive_learning(xtrain,ytrain,xact,yact,n)
simp = simple_margin(xtrain,ytrain,xact_sort,yact_sort,n)
kmed = kmedoids_active_learning(xtrain,ytrain,xact_sort,yact_sort,cut,n)        

xax = np.linspace(1,n,num=n)
plt.figure()
plt.plot(xax,pasi,label="passive",color="g")
plt.plot(xax,simp,label="simple_margin",color="b")
plt.plot(xax,kmed,label="k-mediods",color="r")
plt.legend(loc="best")
plt.show
