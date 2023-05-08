import numpy as np
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
            switch_idx=(prbs_updated<prbs_switched)           # where switching is more likely (vectorized)
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
def gibbs_gmm(y,w,tau,tau0,steps,burnin_pct=0.25,verbose=False,seed=0):
    """
    Run a Gibbs sampler for the means mu and labels x of
    a Gaussian mixture model with obsevations
    yn~sum_k wk Gaussian(muk,tauk), n=1,...,N
    and known weights w and precisions tau

    Inputs:
        y          : (N,) array, observations
        w          : (K,) array, weights
        tau        : (K,) array, precisions
        tau0       : float, prior precision for the mean
        steps      : int, number of steps to run the sampler from (after burn-in)
        burnin_pct : float, percentage of burn-in desired
        verbose    : boolean, indicates whether to print messages
        seed       : int, random seed

     Outputs:
         xs        : (N,steps) array, labels samples
         mus       : (K,steps) array, means samples

    Note: the total number of steps the sampler is run for is
          (T=steps+burn_in), where (burn_in=T*burnin_pct)
          the total burn-in steps is therefore
          (steps*burnin_pct/(1-burnin_pct))
    """
    np.random.seed(0+seed)
    N=y.shape[0]
    K=w.shape[0]
    burnin_steps=int(steps*burnin_pct/(1-burnin_pct))

    # generate initial values
    x=np.random.randint(low=0,high=K,size=N)
    mu=np.random.randn(K)/np.sqrt(tau0)
    if steps==0: return x,mu

    # do burn-in pass
    for t in range(burnin_steps):
        if verbose: print('Burn-in: '+str(t+1)+'/'+str(burnin_steps),end='\r')
        #print(x.shape,mu.shape)
        x,mu=gibbs_gmm_update(x,mu,y,w,tau,tau0)
    # end for

    # do sampling pass
    xs=np.zeros((N,steps+1),dtype=int)
    xs[:,0]=x
    mus=np.zeros((K,steps+1))
    mus[:,0]=mu
    for t in range(steps):
        if verbose: print('Sampling: '+str(t+1)+'/'+str(steps),end='\r')
        tmpx,tmpmu=gibbs_gmm_update(xs[:,t],mus[:,t],y,w,tau,tau0)
        xs[:,t+1]=tmpx
        mus[:,t+1]=tmpmu
    # end for

    # sort according to latest mu
    sort_idx=np.argsort(mus[:,-1])
    mus=mus[sort_idx,:]
    xsc=np.zeros(xs.shape)
    for k in range(K): xsc[xs==k]=sort_idx[k]

    return xsc[:,1:],mus[:,1:]


def gibbs_gmm_update(x,mu,y,w,tau,tau0):
    """
    Do a single pass of a Gibbs sampler for a Gaussian mixture model
    starting at x and mu

    Inputs:
        x          : (N,) array, current labels
        mu         : (K,) array, current means
        y          : (N,) array, observations
        w          : (K,) array, weights
        tau        : (K,) array, precisions
        tau0       : float, prior precision for the mean

     Outputs:
         x        : (N,) array, updated labels
         mu       : (K,) array, updated means
    """
    N=x.shape[0]
    K=w.shape[0]

    # update x probs
    x_probs=w[np.newaxis,:]*np.exp(gauss_lp_allmeans(y,mu,tau))
    x_probs[x_probs<1e-32]=1e-32
    x_probs=x_probs/np.sum(x_probs,axis=1)[:,np.newaxis]

    # sample x via direct inversion (to avoid for loops or np.random.choice)
    Fx=np.cumsum(x_probs,axis=1)
    u=np.random.rand(N,1)
    x_=np.argmax(Fx>u,axis=1)

    # sample mu
    idx=(x_==np.arange(0,K,dtype=int)[:,np.newaxis])
    N_pool=np.sum(idx,axis=1)
    tau_pool=N_pool+tau0
    y_pool=np.sum(y*idx,axis=1)/tau_pool
    mu_=y_pool+np.random.randn(K)/np.sqrt(tau_pool)

    return x_,mu_


def gauss_lp_allmeans(y,mu,tau):
    """
    Evaluate a Gaussian log pdf
    This function evaluates phi(yn;muk,tauk)
    for all combinations of n=1,...,N and k=1,...,K

    Inputs:
        y          : (N,) array, observations
        mu         : (K,) array, means
        tau        : (K,) array, precisions

     Outputs:
         x        : (N,K) array, log pdfs
    """
    return -0.5*((y[:,np.newaxis]-mu[np.newaxis,:])**2)*tau[np.newaxis,:]-0.5*np.log(2*np.pi/tau[np.newaxis,:])
