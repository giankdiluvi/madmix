import numpy as np
from aux import *
from concrete import idx_unflattenBinary, idx_flattenBinary

"""
########################
########################
#    mean field 1D     #
########################
########################
"""
def meanfield1D(lq0,lp,max_iters,gamma):
    """
    Discrete mean field VI for 1D distributions

    Inputs:
        lq0       : (K,) array, initial variational log probabilities
        lp        : (K,) array, target log probabilities
        max_iters : int, max number of gradient ascent iterations
        gamma     : function, step size (as function)
    """
    q=np.exp(lq0)
    for t in range(max_iters): q=q-gamma(t)*(np.log(q)-lp)
    return np.log(q)


"""
########################
########################
#    mean field 2D     #
########################
########################
"""
def meanfield2D(lq1,lq2,lp,max_iters):
    """
    Discrete mean field VI for 2D distributions

    Inputs:
        lq1       : (K1,) array, initial variational log probabilities of X1
        lq2       : (K2,) array, initial variational log probabilities of X2
        lp        : (K1,K2) array, target log probabilities
        max_iters : int, max number of gradient ascent iterations

    Outputs:
        lq1_ : (K1,) array, updated X1 variational log probabilities
        lq2_ : (K2,) array, updated X2 variational log probabilities
    """

    lq1_=np.copy(lq1)
    lq2_=np.copy(lq2)
    for t in range(max_iters):
        # update q1
        lq1_=np.sum(np.exp(lq2_)[np.newaxis,:]*lp,axis=1)
        lq1_=lq1_-LogSumExp(lq1_[:,np.newaxis]) # normalizing constant

        # update q2
        lq2_=np.sum(np.exp(lq1_)[:,np.newaxis]*lp,axis=0)
        lq2_=lq2_-LogSumExp(lq2_[:,np.newaxis]) # normalizing constant
    # end for
    return lq1_,lq2_


"""
########################
########################
#   mean field Ising   #
########################
########################
"""
def meanfieldIsing(M,beta,max_iters):
    """
    Discrete mean field VI for the Ising model
    Warning: only use for M<10

    Inputs:
        M         : int, number of particles
        beta      : float, inverse temperature (beta>0)
        max_iters : int, max number of gradient ascent iterations

    Outputs:
        lq        : (M,2) array, variational log pmf of each q_m
    """

    lq=-np.log(2)*np.ones((M,2)) # initialize at exactly 1/2 each
    for t in range(max_iters):
        for m in range(M):
            # edge cases first
            if m==0: tmplq1=beta*(2*lq[m+1,0]-1) # P(Xm=+1)+c
            if m==M-1: tmplq1=beta*(2*lq[m-1,0]-1) # P(Xm=+1)+c

            # non-edge cases now
            if m>0 and m<M-1: tmplq1=2*beta*(lq[m-1,0]+lq[m+1,0]-1) # P(Xm=+1)+c
            tmplq=np.array([-1.*tmplq1,tmplq1]) # (P(Xm=-1),P(Xm=+1))+c
            lZ=LogSumExp(tmplq[:,np.newaxis])
            lq[m,:]=tmplq-lZ
        # end for
    # end for

    return lq

def flattenlq(lq):
    M=lq.shape[0]
    tmpx=idx_unflattenBinary(np.arange(2**M),M)
    lq_flat=np.zeros(2**M)
    for i in range(2**M): lq_flat[i]=np.sum(lq[np.arange(M),tmpx[:,i]])
    return lq_flat
