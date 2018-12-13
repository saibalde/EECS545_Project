#Author: David Dang
#Class: EECS545 Machine Learning
#Title: TSA Implementation
#Date: 11-06-2018

import numpy as np
from graph import subarray
import time

def query_with_TSA(graph):
   TSA_instance = TSA(graph)
   TSA_instance.calc_lookahead_risk()

   q = TSA_instance.solve_eem()
   #print('q',q)

   return q;

#function for removing i-th row and column from arr (NxN); returns matrix that is
# (N-1)x(N-1)
def cut_row_col(arr,i):
    N = arr.shape[0]
    mat_cut = np.empty((N-1,N-1), arr.dtype)
    mat_cut[:i, :i] = arr[:i, :i]        #upper left block
    mat_cut[:i, i:] = arr[:i, i+1:]      #upper right block
    mat_cut[i:, :i] = arr[i+1:, :i]      #lower left block
    mat_cut[i:, i:] = arr[i+1:, i+1:]    #lower right block
    return mat_cut   

"""
Class for Two-Step Approximation (The Query Step)
"""
class TSA:
    def __init__(self, graph):
        #does this require operator= ?
        self.graph = graph          #may change later from composition to inheritance
        len_u = len(self.graph.u)
        self.set_y_ell(graph.labels[graph.l])
        self.f = np.zeros([len_u,1])
        self.lookahead_risk = np.zeros(len_u)
        self.marginals = np.zeros(len_u)
        
        
    '''
    -- Eq. (10) from TSA; variables here named accordingly
    -- Function for computing f_k, the decision value of node k s.t. k \in u
    -- After computing f, we must get marginals, P(Y_k=1|Y_\ell = y_\ell) ~ sigma(f_k)
       for each k 
    -- y_ell are labels for ell; y_ell[i] corresponds to index ell[i]
    -- toggle is true if computing marginals in Eq. (7); false otherwise
    '''
    def calc_marginals(self, y_ell, toggle, idx_2_remove=-1, u_excluding_q=-1, ell_with_q=-1):
        #trick; this is also the var G in Appendix A.3
        # self.L_uu_inv = np.linalg.inv(L_uu) 

        if toggle:
            laplacian_uu_inv_kk = self.graph.LuuInv.diagonal()
            laplacian_uu_inv_kk = laplacian_uu_inv_kk.reshape([len(laplacian_uu_inv_kk),1])

            y_ell = np.reshape(y_ell,(len(y_ell),1))
            f = np.multiply(-2.0/laplacian_uu_inv_kk, \
                 np.matmul(np.matmul(self.graph.LuuInv,self.graph.laplacian_ul()),y_ell)) 

        else:
            print('this should never be executed if using the dongle trick')
            # exit()
            laplacian_uu = self.graph.laplacian[u_excluding_q,:][:,u_excluding_q]
            laplacian_uu_inv = np.linalg.inv(laplacian_uu)
            laplacian_uu_inv_kk = laplacian_uu_inv.diagonal()
            laplacian_uu_inv_kk = laplacian_uu_inv_kk.reshape([len(laplacian_uu_inv_kk),1])
            laplacian_ul = self.graph.laplacian[u_excluding_q,:][:,ell_with_q]

            f = np.multiply(-2.0/laplacian_uu_inv_kk, \
                 np.matmul(np.matmul(laplacian_uu_inv,laplacian_ul),y_ell))
                  
        if toggle:
            self.f = f
            
        marginals = 1.0/(1+np.exp(-f))

        return marginals
    
    
    '''
    -- Function for computing f using the dongle trick; see last eqn of TSA paper
    -- i here is for the queried node?
    -- This function should be called instead of calc_marginals after the 1st
       query
    '''
    def dongle_trick(self,i):
        t0 = time.time()
        diag_idx = np.arange(self.graph.LuuInv.shape[0])
        diag_idx_wo_i = np.delete(diag_idx,i)
        G_kk = self.graph.LuuInv.diagonal()
        G_kk = np.delete(G_kk,i)
        G_kk = G_kk.reshape([len(G_kk),1])
        G_ki = self.graph.LuuInv[:,i]
        G_ki = np.delete(G_ki,i,0)
        # G_ki = self.graph.LuuInv[diag_idx_wo_i,i]
        # G_ki = G_ki.reshape([len(G_ki),1])
        G_ii = self.graph.LuuInv[i,i]
        #f_kk = self.f[self.u]
        f_i = self.f[i]

        #y0[i] corresponds to the y_0 in the dongle trick on appendix
        y0 = np.zeros((2,1))
        y0[0] = 1
        y0[1] = -1

        #The term in big [] in App. A.3
        left_hadamard = 1/(G_kk - np.square((G_ki).reshape([len(G_ki),1]))/G_ii)  

        #The term in big () in App. A.3
        right_hadamard = np.multiply(0.5*G_kk,self.f[diag_idx_wo_i]) + \
            np.matrix.transpose(np.multiply((y0/G_ii - f_i/2),G_ki))

        #LHS is [f_k^{+i}]_{k \in \bar{u}} after observing Y_{\ell \union \{i\}}
        f = 2*np.multiply(left_hadamard,right_hadamard)
        marginals = 1.0/(1+np.exp(-f))

        return marginals
    
    
    '''    
    -- Eq. (6) from TSA paper
    -- Function for compute zero-one risk
    -- q is the index of the queried node
    -- y_ell_with_q is the labels for ell and q
    -- ell_with_q is the set of indices for ell and q
    '''
    # def calc_zero_one_risk(self,q,y_ell_with_q,ell_with_q,u_excluding_q):
    def calc_zero_one_risk(self,i):
        zero_one_risk = np.zeros(2)
        
        '''
        Compute marginals in Eq. 6; note that it is computed for the entire
        set u_excluding_q simultaneously; also note this will output a 2 column 
        vector because Y_q here is a 2 column vector. The columns of Y_q are 
        identical except for the first element which is 1 & -1 in the 1st&2nd 
        columns, respectively
        '''
        # marginals = self.calc_marginals(y_ell_with_q, False, idx_2_remove, u_excluding_q, ell_with_q)
        marginals = self.dongle_trick(i)
        # print(marginals.shape)

        #note that zero_one_risk is going to be a 1x2 array
        #the first column corresponds to Y_q = [1,y_ell] and the second to
        # Y_q = [-1,y_ell]
        # print(marginals.shape)
        # t0 = time.time()
        # num_rows, num_cols = marginals.shape
        # for i in range(0,num_rows):
        #     if marginals[i,0] >= 0.5:
        #         zero_one_risk[0] += 1 - marginals[i,0]
        #     else:
        #         zero_one_risk[0] += marginals[i,0]                
        #     if marginals[i,1] >= 0.5:
        #         zero_one_risk[1] += 1 - marginals[i,1]
        #     else:
        #         zero_one_risk[1] += marginals[i,1]        
        
        #these 2 lines vectorize the above for loop
        zero_one_risk[0] = np.sum(np.minimum(marginals[:,0], 1-marginals[:,0]))
        zero_one_risk[1] = np.sum(np.minimum(marginals[:,1], 1-marginals[:,1]))

        # t1 = time.time()
        # print(t1-t0)    
        zero_one_risk *= 1.0/self.graph.num_nodes
        
        return zero_one_risk 
        
    
    '''
    -- Eq. (7) from TSA paper    
    -- Compute lookahead zero-one risk
    -- y_ell is the labels for ell
    -- q is the queried node's index
    '''
    def calc_lookahead_risk(self):
        y_ell_with_q = np.zeros([len(self.y_ell)+1,2])
        y_ell_with_q[:,0] = np.append(self.y_ell,1)
        y_ell_with_q[:,1] = np.append(self.y_ell,-1)

        #computes marginals for all q \in u. This is the marginal prob. in Eq.(7)
        self.marginals = self.calc_marginals(self.y_ell,True)  #marginals in Eq. (7)

        for i in range(0,len(self.graph.u)):
            zero_one_risk = self.calc_zero_one_risk(i)

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

        # print('q',q)

        return q
        
    
    def set_y_ell(self,y_ell):
        assert type(y_ell) == np.ndarray

        self.y_ell = y_ell        
