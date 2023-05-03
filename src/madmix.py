import numpy as np
import scipy.stats as stats
import pandas as pd
import discrete_mixflows as dmfs
import ham_mixflows as cmfs
from aux import *



"""
########################################
########################################
variational approximation functions
########################################
########################################
"""

def lqN(xd,ud,xc,rho,uc,N,lq0,L,epsilon,lp,grad_lp,xi=np.pi/16):
    """
    evaluate variational log likelihood log qN(xd,ud,xc,rho,uc)

    inputs:
        xd        : (M,d) array, initial discrete states
        ud        : (M,d) array, initial discrete pseudo times
        xc        : (M,d) array, initial positions
        rho       : (M,d) array, initial momenta
        uc        : (d,)  array, initial continuous pseudo times
        N         : int, variational parameter; max number of applications of T
        lq0       : function, reference log likelihood (vectorized)
        L         : int, number of leapfrog steps for Hamiltonian dynamics
        epsilon   : float, step size for Hamiltonian dynamics
        lp        : function, posterior and conditional pmf
        grad_lp   : function, target score function generator for Hamiltonian dynamics (vectorized)
        xi        : scalar, pseudo time shift

     outputs:
         log qN(xd,ud,xc,rho,uc) : (d,) array, likelihood at each point
    """
    # save copies to prevent changing input arrays
    xd_,ud_,xc_,rho_,uc_=np.copy(xd),np.copy(ud),np.copy(xc),np.copy(rho),np.copy(uc)

    # define laplacian momentum fns
    lm,Fm,Qm,grad_lm=cmfs.lap_lm,cmfs.lap_Fm,cmfs.lap_Qm,cmfs.lap_gradlm

    if N==0: return lq0(xd_,ud_,xc_,rho_,uc_) # no steps taken here

    # init weights and log jacobians
    w=np.zeros((N,xc_.shape[1]))
    w[0,:]=lq0(xd_,ud_,xc_,rho_,uc_)
    lJ=np.zeros(xc_.shape[1])

    # iterate through flow
    for n in range(N-1):
        xd_,ud_,xc_,rho_,uc_,tlJ=flow(1,xd_,ud_,xc_,rho,uc_,L,epsilon,lp,grad_lp,lm,Fm,Qm,grad_lm,xi,'bwd') # one step bwd
        lJ=lJ+tlJ # update log jacobian
        w[n+1,:]=lq0(xd_,ud_,xc_,rho_,uc_)+tlJ # update weight
    # end for
    return LogSumExp(w)-np.log(N)


def randqN(size,N,randq0,L,epsilon,lp,grad_lp,xi=np.pi/16):
    """
    generate samples from the variational distribution qN

    inputs:
        size      : int, number of samples to generate
        N         : int, variational parameter; max number of applications of T
        randq0    : function, reference distribution sampler
        L         : int, number of leapfrog steps for Hamiltonian dynamics
        epsilon   : float, step size for Hamiltonian dynamics
        lp        : function, posterior and conditional pmf
        grad_lp   : function, target score function generator for Hamiltonian dynamics (vectorized)
        xi        : scalar, pseudo time shift

     outputs:
        rx      : (M,size) array, position samples
        rrho    : (M,size) array, momentum samples
        ru      : (size,) array, pseudo time samples
    """

    if N==1: return randq0(size)                     # sample from reference directly

    K=np.random.randint(low=0,high=N,size=size)      # generate number of steps per sample point
    lm,Fm,Qm,grad_lm=cmfs.lap_lm,cmfs.lap_Fm,cmfs.lap_Qm,cmfs.lap_gradlm # laplacian momentum
    xd,ud,xc,rho,uc=randq0(size)                             # initialize samples
    for n in range(N):
        print('Sampling '+str(n+1)+'/'+str(N),end='\r')
        # update points where the current n does not exceed their corresponding K
        idx=K>=n+1
        #if np.sum(idx)==0: continue # if all values have sampled K < n then don't take more steps
        txd,tud,txc,trho,tuc,_=flow(1,xd[:,idx],ud[:,idx],xc[:,idx],rho[:,idx],uc[idx],L,epsilon,lp,grad_lp,lm,Fm,Qm,grad_lm,xi,'fwd')

        # save updates
        xd[:,idx]=txd
        ud[:,idx]=tud
        xc[:,idx]=txc
        rho[:,idx]=trho
        uc[idx]=tuc
    # end for
    return xd,ud,xc,rho,uc


"""
########################################
########################################
flow functions
########################################
########################################
"""

def flow(steps,xd,ud,xc,rho,uc,L,epsilon,lp,grad_lp,lm,Fm,Qm,grad_lm,xi,direction):
    """
    compute T^n(xd,ud,xc,rho,uc)

    inputs:
        steps     : int, number of applications of T, n
        xd        : (M,d) array, initial discrete states
        ud        : (M,d) array, initial discrete pseudo times
        xc        : (M,d) array, initial positions
        rho       : (M,d) array, initial momenta
        uc        : (d,)  array, initial continuous pseudo times
        L         : int, number of leapfrog steps for Hamiltonian dynamics
        epsilon   : float, step size for Hamiltonian dynamics
        lp        : function, posterior and conditional pmf
        grad_lp   : function, target score function generator for Hamiltonian dynamics (vectorized)
        lm,Fm,Qm,grad_lm : functions, momentum log density, cdf, quantile fn, and score fn (respectively)
        xi        : scalar, pseudo time shift
        direction : string, one of 'fwd' (forward map) or 'bwd' (backward map)

     outputs:
       x'   : (M,d) array, updated positions
       rho' : (M,d) array, updated momenta
       u'   : (M,d) array, updated pseudo times
       ljs  : (d,)  array, log Jacobians for each sample point
    """
    # save copies to prevent changing input arrays
    xd_,ud_,xc_,rho_,uc_=np.copy(xd),np.copy(ud),np.copy(xc),np.copy(rho),np.copy(uc)

    # initialize log jacobians and return input if no steps
    lJ=np.zeros(xc_.shape[1])
    if steps==0: return xd_,ud_,xc_,rho_,uc_,lJ

    # iterate flow
    for n in range(steps):
        # take a step and update log jacobian
        curr_grad_lp=grad_lp(xd_) # define new grad lp
        if direction=='fwd': xd_,ud_,xc_,rho_,uc_,tmplJ=forward(xd_,ud_,xc_,rho_,uc_,L,epsilon,lp,curr_grad_lp,lm,Fm,Qm,grad_lm,xi)
        if direction=='bwd': xd_,ud_,xc_,rho_,uc_,tmplJ=backward(xd_,ud_,xc_,rho_,uc_,L,epsilon,lp,curr_grad_lp,lm,Fm,Qm,grad_lm,xi)
        lJ=lJ+tmplJ
    # end for
    return xd_,ud_,xc_,rho_,uc_,lJ


def forward(xd,ud,xc,rho,uc,L,epsilon,lp,grad_lp,lm,Fm,Qm,grad_lm,xi):
    """
    compute T(xd,ud,xc,rho,uc) (i.e., one step forward)

    inputs:
        xd        : (M,d) array, initial discrete states
        ud        : (M,d) array, initial discrete pseudo times
        xc        : (M,d) array, initial positions
        rho       : (M,d) array, initial momenta
        uc        : (d,)  array, initial continuous pseudo times
        L         : int, number of leapfrog steps for Hamiltonian dynamics
        epsilon   : float, step size for Hamiltonian dynamics
        lp        : function, posterior and conditional pmf
        grad_lp   : function, target score function for Hamiltonian dynamics (vectorized)
        lm,Fm,Qm,grad_lm : functions, momentum log density, cdf, quantile fn, and score fn (respectively)
        xi        : scalar, pseudo time shift

     outputs:
         xd'   : (M,d) array, updated discrete states
         ud'   : (M,d) array, updated discrete pseudotimes
         xc'   : (M,d) array, updated positions
         rho'  : (M,d) array, updated momenta
         uc'   : (M,d) array, updated continuous pseudo times
         lJs   : (d,)  array, log Jacobians
    """
    # take step in continuous space
    xc_,rho_,uc_,lJc=cmfs.forward(xc,rho,uc,L,epsilon,grad_lp,lm,Fm,Qm,grad_lm,xi)

    # take step in discrete space
    xd_,ud_=np.copy(xd),np.copy(ud)
    M=xd_.shape[0]
    lJd=np.zeros(xd_.shape)
    for m in range(M):
        tmp_prbs=np.atleast_1d(np.exp(lp(xd_,xc_,axis=m)))
        tmp_prbs=tmp_prbs/np.sum(tmp_prbs,axis=1)[:,np.newaxis]
        tx,tu,tlJd=dmfs.Tm(xd_[m,:],ud_[m,:],tmp_prbs,xi,direction='fwd')
        xd_[m,:]=tx
        ud_[m,:]=tu
        lJd[m,:]=tlJd
    # end for
    return xd_,ud_,xc_,rho_,uc_,lJc+lJd


def backward(xd,ud,xc,rho,uc,L,epsilon,lp,grad_lp,lm,Fm,Qm,grad_lm,xi):
    """
    compute T^{-1}(xd,ud,xc,rho,uc) (i.e., one step backward)

    inputs:
        xd        : (M,d) array, initial discrete states
        ud        : (M,d) array, initial discrete pseudo times
        xc        : (M,d) array, initial positions
        rho       : (M,d) array, initial momenta
        uc        : (d,)  array, initial continuous pseudo times
        L         : int, number of leapfrog steps for Hamiltonian dynamics
        epsilon   : float, step size for Hamiltonian dynamics
        lp        : function, posterior and conditional pmf
        grad_lp   : function, target score function for Hamiltonian dynamics (vectorized)
        lm,Fm,Qm,grad_lm : functions, momentum log density, cdf, quantile fn, and score fn (respectively)
        xi        : scalar, pseudo time shift

     outputs:
         xd'   : (M,d) array, updated discrete states
         ud'   : (M,d) array, updated discrete pseudotimes
         xc'   : (M,d) array, updated positions
         rho'  : (M,d) array, updated momenta
         uc'   : (M,d) array, updated continuous pseudo times
         lJs   : (d,)  array, log Jacobians
    """
    # take step in continuous space
    xc_,rho_,uc_,lJc=cmfs.backward(xc,rho,uc,L,epsilon,grad_lp,lm,Fm,Qm,grad_lm,xi)

    # take step in discrete space
    lJd=np.zeros(xd.shape)
    M=xd.shape[0]
    xd_,ud_=np.copy(xd),np.copy(ud)
    for m in reversed(range(M)):
        tmp_prbs=np.atleast_1d(np.exp(lp(xd_,xc_,axis=m)))
        tmp_prbs=tmp_prbs/np.sum(tmp_prbs,axis=1)[:,np.newaxis]
        tx,tu,tlJd=dmfs.Tm(xd_[m,:],ud_[m,:],tmp_prbs,xi,direction='bwd')
        xd_[m,:]=tx
        ud_[m,:]=tu
        lJd[m,:]=tlJd
    # end for
    return xd_,ud_,xc_,rho_,uc_,lJc+np.sum(lJd,axis=0)
