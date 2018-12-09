"""
Author: David Dang
Class: EECS545 Machine Learning
Title: TSA Implementation
Date: 11-06-2018
"""

import numpy as np
from util import subarray


def query_with_TSA(graph):
   TSA_instance = TSA(graph)
   TSA_instance.calc_lookahead_risk()
   q = TSA_instance.solve_eem()

   return q;

"""
Class for Two-Step Approximation (The Query Step)
"""
class TSA:
    # def __init__(self, ell, u, y_ell, graph):
    def __init__(self, graph):
        #does this require operator= ?
        len_u = len(u)
        self.graph = graph          #may change later from composition to inheritance
        # self.set_ell(ell)           #ell is a set of labeled indices
        # self.set_u(u)               #u is a set of unlabeled indices
        self.set_y_ell(graph.labels[graph.l])
        # self.L_uu_inv = np.zeros([len_u,len_u])
        self.f = np.zeros([len_u,1])
        self.lookahead_risk = np.zeros(len_u)
        self.marginals = np.zeros(len_u)
        
        
    
    '''
    -- Eq. (10) from TSA; variables here named accordingly
    -- Function for computing f_k, the decision value of node k s.t. k \in u
    -- After computing f, we must get marginals, P(Y_k=1|Y_\ell = y_\ell) ~ sigma(f_k)
       for each k 
    -- y_ell are labels for ell; y_ell[i] corresponds to index ell[i]
    -- toggle is true if computingmarginals in Eq. (7); false otherwise
    -- Need to make ell and y_ell optional arguments
    '''
    def calc_marginals(self, ell, y_ell, u, toggle):
        # L_uu = subarray(self.graph.laplacian,u,u)
        self.graph.laplacian_ul = subarray(self.graph.laplacian,u,ell)
        
        #make this member variable because we will use this for the dongle
        #trick; this is also the var G in Appendix A.3
        # self.L_uu_inv = np.linalg.inv(L_uu) 
        laplacian_uu_inv_kk = self.graph.laplacian_uu_inv.diagonal()
        laplacian_uu_inv_kk = laplacian_uu_inv_kk.reshape([len(laplacian_uu_inv_kk),1])
        
        f = np.multiply(-2.0/laplacian_uu_inv_kk, \
                             np.matmul(np.matmul(self.graph.laplacian_uu_inv,self.graph.laplacian_ul),y_ell)) 
        if toggle:
            self.f = f
            
        marginals = 1.0/(1+np.exp(f))
        
        return marginals
    
    
    
    
    '''
    -- Function for computing f using the dongle trick; see last eqn of TSA paper
    -- i here is
    -- how do we know what y0 is?
    -- This function should be called instead of calc_marginals after the 1st
       query
    '''
    def dongle_trick(self,i,y_0):
        G_diag = self.graph.laplacian_uu_inv.diagonal()  
        G_kk = G_diag                    #note G_kk is actually [G_kk]_k, a set
        # G_ki = self.graph.laplacian_uu_inv[self.u,i]
        G_ki = self.graph.laplacian_uu_inv[self.graph.u,i]
        G_ii = self.graph.laplacian_uu_inv[i,i]
        #f_kk = self.f[self.u]
        f_i = self.f[i]

        left_hadamard = 1/(G_kk - np.square(G_ki)/G_ii)
        right_hadamard = np.multiply(0.5*G_diag,self.f) + \
            (y_0/G_ii - f_i/2)*self.graph.laplacian_uu_inv[:,i]

        #LHS is [f_k^{+i}]_{k \in \bar{u}} after observing Y_{\ell \union {i}}
        self.f = 2*np.multply(left_hadamard,right_hadamard)
        marginals = 1.0/(1+np.exp(-self.f))

        return marginals
    
    
    
    '''    
    -- Eq. (6) from TSA paper
    -- Function for compute zero-one risk
    -- q is the index of the queried node
    -- y_ell_q is the labels for ell and q
    -- ell_q is the set of indices for ell and q
    '''
    def calc_zero_one_risk(self,q,y_ell_q,ell_q):
        zero_one_risk = np.zeros(2)
        
        #remove q from u and store in u_excluding_q
        u_excluding_q = np.setdiff1d(self.graph.u,q) 
        
        '''
        Compute marginals in Eq. 6; note that it is computed for the entire
        set u_excluding_q simultaneously; also note this will output a 2 column 
        vector because Y_q here is a 2 column vector. The columns of Y_q are 
        identical except for the first element which is 1 & -1 in the 1st&2nd 
        columns, respectively
        '''
        marginals = self.calc_marginals(ell_q,y_ell_q,u_excluding_q,False) 

        #note that zero_one_risk is going to be a 1x2 array
        #the first column corresponds to Y_q = [1,y_ell] and the second to
        #Y_q = [-1,y_ell]
        #print(marginals.shape)
        num_rows, num_cols = marginals.shape
        for i in range(0,num_rows):
            if marginals[i,0] >= 0.5:
                zero_one_risk[0] += 1 - marginals[i,0]
            else:
                zero_one_risk[0] += 1 - (1 - marginals[i,0])                
            if marginals[i,1] >= 0.5:
                zero_one_risk[1] += 1 - marginals[i,1]
            else:
                zero_one_risk[1] += 1 - (1 - marginals[i,1])        
            
        zero_one_risk *= 1.0/self.graph.num_nodes
        
        return zero_one_risk 
        
    
    
    '''
    -- Eq. (7) from TSA paper    
    -- Compute lookahead zero-one risk
    -- y_ell is the labels for ell
    -- q is the queried node's index
    '''
    def calc_lookahead_risk(self):
        y_ell_q = np.zeros([len(self.y_ell)+1,2])
        y_ell_q[:,0] = np.append(self.y_ell,1)
        y_ell_q[:,1] = np.append(self.y_ell,-1)
        
        #computes marginals for all q \in u. This is the marginal in Eq.(7)
        self.marginals = self.calc_marginals(self.graph.l,self.y_ell,self.graph.u,True)  #marginals in Eq. (7)
        
        for i in range(0,len(self.graph.u)):
            q = self.graph.u[i]        #var for storing queried node idx
            
            ell_q = np.append(self.graph.l,q)   #ell with addition of q
            y_ell_q = np.zeros((ell_q.size,2))     #var for storing labels for ell & q

            #compute zero-one risk
            zero_one_risk = self.calc_zero_one_risk(q,y_ell_q,ell_q)
        
            #note this computes marginals for all of u
            #how do i write an optional arg for the below?

            #Compute lookahead_risk for all of q \in u
            #look_ahead risk should be a (u.size,) size 1-D array
            
            self.lookahead_risk[i] = zero_one_risk[0] * self.marginals[i] + \
                zero_one_risk[1] * (1 - self.marginals[i])
        
        
        
    '''
    -- Eq.(5) from TSA paper
    -- Finds the query q that minimizes the lookahead zero one risk
    '''    
    def solve_eem(self):
        idx = np.argmin(self.lookahead_risk)
        q = self.graph.u[idx]
        
        # if self.marginals[idx]>=0.5:
        #     y_q = 1
        # else:
        #     y_q = -1
        
        return q#, y_q
        
    
    
    #def set_idx:
    
    
    def set_y_ell(self,y_ell):
        assert type(y_ell) == np.ndarray

        self.y_ell = y_ell        


# TSAobj = TSA(Z,beta,numSamples)
# TSAobj.calcLaplacian(numSamples,weights)
# TSAobj.calcProbability(labels)

"""
ell_1 is initial labeled nodes. 

Suppose we have observed the labels of nodes \ell as y_{\ell}. 
TSAA to poterior marignal distribution P_{Y_{\ell}}(Y_k) -> query algorithm
\mu is two-step upperbound on P(Y_k=y_k, Y_\ell = y_\ell)


ell: labeled nodes
u_bar: unlabeled nodes except k (u\{k})
u: set of unlabeled nodes
k: unlabeled node excluded from u_bar
"""
