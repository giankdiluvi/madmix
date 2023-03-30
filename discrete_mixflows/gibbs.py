import numpy as np
from discrete_mixflows import *


def gibbs_sampler(x0,steps,lp,burnin_pct=0.25,verbose=False):
    """
    run a Gibbs sampler targeting exp(lp)
    
    inputs:
        x0         : (M,) array, initial states of x
        steps      : int, number of steps to run the sampler from (after burn-in)
        lp         : function, target log likelihood (vectorized)
        burnin_pct : float, percentage of burn-in desired
        verbose    : boolean, indicates whether to print messages
        
     outputs:
         xs        : (M,steps) array, Gibbs samples
         
    note: the total number of steps the sampler is run for is 
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
        xs[:,t+1]=gibbs_update(xs[:,t],lp)
    # end for
    return xs[:,1:]


def gibbs_update(x,lp):
    """
    do a single pass of a Gibbs sampler targeting exp(lp) starting at x
    
    inputs:
        x  : (M,) array, initial state of x
        lp : function, target log likelihood (vectorized)
        
     outputs:
         x'        : (M,) array, state after a Gibbs pass
    """
    
    M=x.shape[0]
    for m in range(M):
        prbs_m=np.squeeze(np.exp(lp(x[:,np.newaxis],axis=m)))
        prbs_m=prbs_m/np.sum(prbs_m)
        x[m]=np.random.choice(a=np.arange(0,prbs_m.shape[0]),p=prbs_m)
    # end for
    return x