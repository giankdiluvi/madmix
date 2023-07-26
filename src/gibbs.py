import numpy as np
import sys
from discrete_mixflows import *

"""
########################################
########################################
discrete variable inference only
########################################
########################################
"""
def gibbs_sampler(x0,steps,lp,burnin_pct=0.25,switch=False,verbose=False):
    """
    Run a Gibbs sampler targeting exp(lp)

    Inputs:
        x0         : (M,) array, initial states of x
        steps      : int, number of steps to run the sampler from (after burn-in)
        lp         : function, target log likelihood (vectorized)
        burnin_pct : float, percentage of burn-in desired
        switch     : boolean, indicates whether to propose the switch x -> 1-x for ising model
        verbose    : boolean, indicates whether to print messages

     Outputs:
         xs        : (M,steps) array, Gibbs samples

    Note: the total number of steps the sampler is run for is
          (T=steps+burn_in), where (burn_in=T*burnin_pct)
          the total burn-in steps is therefore
          (steps*burnin_pct/(1-burnin_pct))
    """

    if steps==0: return x0[:,np.newaxis]
    burnin_steps=int(steps*burnin_pct/(1-burnin_pct))

    # burn-in pass
    for t in range(burnin_steps):
        if verbose: print('Burn-in: '+str(t+1)+'/'+str(burnin_steps),end='\r')
        x0=gibbs_update(x0,lp)
    # end for

    # sampling pass
    xs=np.zeros((x0.shape[0],steps+1),dtype=int)
    xs[:,0]=x0
    for t in range(steps):
        if verbose: print('Sampling: '+str(t+1)+'/'+str(steps),end='\r')
        x_updated=gibbs_update(xs[:,t],lp)
        if switch:
            prbs_updated=np.squeeze(np.exp(lp(x_updated[np.newaxis,:])))    # prbs if we don't switch
            prbs_switched=np.squeeze(np.exp(lp(1-x_updated[np.newaxis,:]))) # prbs if we do switch
            switch_idx=(prbs_switched/prbs_updated)>np.random.rand(x0.shape[0])  # where switching is more likely (vectorized)
            x_updated[switch_idx]=1-x_updated[switch_idx]     # update x accordingly
        # end if
        xs[:,t+1]=x_updated
    # end for
    return xs[:,1:]


def gibbs_update(x,lp,switch=False):
    """
    Do a single pass of a Gibbs sampler targeting exp(lp) starting at x

    Inputs:
        x   : (M,) array, initial state of x
        lp  : function, target log likelihood (vectorized)

     Outputs:
         x' : (M,) array, state after a Gibbs pass
    """

    M=x.shape[0]
    for m in range(M):
        prbs_m=np.squeeze(np.exp(lp(x[:,np.newaxis],axis=m)))
        prbs_m=prbs_m/np.sum(prbs_m)
        x[m]=np.random.choice(a=np.arange(0,prbs_m.shape[0]),p=prbs_m)
    # end for
    return x


"""
########################################
########################################
Gaussian mixture model Gibbs sampler
########################################
########################################
"""

def gibbs_gmm(y,mu0,sigma0,w0,steps,burnin_pct,seed=0):
    """
    Run a Gibbs sampler for the labels x, weights w,
    means mu, and covariance matrices sigma of
    a Gaussian mixture model with obsevations
    yn~sum_k wk Gaussian(muk,sigmak), n=1,...,N, yn in R^D

    Inputs:
        y          : (N,D) array, observations (N is no. of obs, d is dimension of each obs)
        mu0        : (K,D) array, initial means (K is number of clusters)
        sigma0     : (K,D,D) array, initial covariances
        w0         : (K,) array, initial weights
        steps      : int, number of steps to run the sampler from (after burn-in)
        burnin_pct : float, percentage of burn-in desired
        seed       : int, random seed

     Outputs:
         xs        : (steps,N) array, labels samples
         ws        : (steps,K) array, weights samples
         mus       : (steps,K,D) array, means samples
         sigmas    : (steps,K,D,D) array, covariance matrices samples

    Note: the total number of steps the sampler is run for is
          (T=steps+burn_in), where (burn_in=T*burnin_pct).
          The total burn-in steps is therefore
          (steps*burnin_pct/(1-burnin_pct))
    """
    np.random.seed(0+seed)

    # get sizes, calculate steps
    N,d=y.shape
    K=mu0.shape[0]
    burnin_steps=int(steps*burnin_pct/(1-burnin_pct))
    total_steps=burnin_steps+steps+1

    # init params
    xs=np.zeros((total_steps,N),dtype=int)
    xs[0,:]=np.random.randint(low=0,high=K,size=N)
    ws=np.ones((total_steps,K))/K
    ws[0,:]=w0
    mus=np.zeros((total_steps,K,d))
    mus[0,:,:]=mu0
    sigmas=np.ones((total_steps,K,d,d))
    sigmas[0,:,:,:]=sigma0

    for t in range(total_steps-1):
        if t<burnin_steps: print('Burn-in: '+str(t+1)+'/'+str(burnin_steps),end='\r')
        if t>=burnin_steps: print('Sampling: '+str(t+1-burnin_steps)+'/'+str(steps),end='\r')

        # update indices ###
        # first obtain log probabilities
        tmplprbs=np.ones((N,K))*np.log(ws[t,:])
        for k in range(K): tmplprbs[:,k]+=stats.multivariate_normal(mus[t,k,:],sigmas[t,k,:,:]).logpdf(y)
        # then sample using gumbel-max trick
        G=np.random.gumbel(size=(N,K))
        tmpx=np.argmax(tmplprbs+G,axis=1)
        xs[t+1,:]=tmpx

        # get cluster summaries
        x_tuple=np.zeros((N,K),dtype=int)
        x_tuple[np.arange(N),tmpx]=1
        Nks=np.sum(x_tuple,axis=0)

        # update weights ###
        tmpw=np.random.dirichlet(Nks+1)
        ws[t+1,:]=tmpw

        # update means and covariances ###
        for k in range(K):
            yk=y[tmpx==k,:] # cluster elements, avg in next line
            Nk=yk.shape[0]

            # update covariance
            Sk=Nk*np.cov(yk,rowvar=False) # cluster covariance
            tmpsigma=sigmas[t,k,:,:]
            if np.linalg.cond(Sk) < 1/sys.float_info.epsilon: tmpsigma = stats.invwishart(Nk-d-1,Sk).rvs() # Sk invertible
            sigmas[t+1,k,:,:]=tmpsigma

            # update mean
            mus[t+1,k,:]=np.random.multivariate_normal(np.mean(yk,axis=0),sigmas[t+1,k,:,:]/Nk)
        # end for
    # end for
    burnin_steps+=1 # to account for initial draw
    return xs[burnin_steps:,...],ws[burnin_steps:,...],mus[burnin_steps:,...],sigmas[burnin_steps:,...]
