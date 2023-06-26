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
def meanfieldIsing(lprbs,max_iters):
    """
    Discrete mean field VI for the Ising model
    Warning: only use for M<10

    Inputs:
        lprbs     : (2**M,) array, target log probabilities
        max_iters : int, max number of gradient ascent iterations
    """
    M=int(np.log2(lprbs.shape))
    #lq=-np.log(2)*np.ones((M,2)) # exactly 1/2 each
    #lq=0.5*np.ones((M,2))
    #lq[0,0]+=0.0001       # 1/2 each + small noise
    lq=np.random.rand(M,2) # random
    lq=np.log(lq/np.sum(lq))


    tmpx=idx_unflattenBinary(np.arange(2**(M-1)),M-1)
    for t in range(max_iters):
        for m in reversed(range(M)):
            tmplp0=lprbs[x[m,:]==0] # select from log p those with xm=0
            tmplp1=lprbs[x[m,:]==1] # select from log p those with xm=1
            tmplq=np.delete(lq,m,axis=0) # rm prbs from xm

            # now calculate all 2*(M-1) possible products qm'*qm'' for expectation
            ljoint0=np.zeros(2**(M-1))
            ljoint1=np.zeros(2**(M-1))
            for i in range(2**(M-1)):
                ljoint0[i]=np.sum(tmplq[np.arange(M-1),tmpx[:,i]])
                ljoint1[i]=np.sum(tmplq[np.arange(M-1),tmpx[:,i]])
            # end for

            # obtain joint and normalize
            tmplq=np.array([np.sum(ljoint0*tmplp0),np.sum(ljoint1*tmplp1)])
            lZ=LogSumExp(tmplq[:,np.newaxis])

            # update joint
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
