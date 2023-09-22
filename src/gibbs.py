import numpy as np
import scipy.stats as stats
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



"""
########################################
########################################
Spike and Slab Gibbs sampler
########################################
########################################
"""

def gibbs_sas(y,x,steps,burnin_pct,seed=0):
    """
    Run a Gibbs sampler for a spike-and-slab model
    with univariate data y and K-dimensional covariates x
    y~N(x.T beta, sigma**2) where beta is sparse.

    Inputs:
        y          : (N,) array, observations
        x          : (N,K) array, covariates (N is no. of observations, K is no. of covariates)
        steps      : int, number of steps to run the sampler from (after burn-in)
        burnin_pct : float, percentage of burn-in desired
        seed       : int, random seed

     Outputs:
         pis     : (steps,K) array, predicted inclusion idx
         betas   : (steps,K) array, predicted regression coefficients
         thetas  : (steps,) array, predicted inclusion prbs
         sigmas2 : (steps,) array, predicted obs variances
         taus2   : (steps,) array, predicted beta variances

    Note: the total number of steps the sampler is run for is
          (T=steps+burn_in), where (burn_in=T*burnin_pct).
          The total burn-in steps is therefore
          (steps*burnin_pct/(1-burnin_pct))
    """
    np.random.seed(0+seed)

    # set hyperparams
    a,b=1.,1.
    a1,a2=0.1,0.1
    s=0.5

    # get sizes, calculate steps
    N,K=x.shape
    burnin_steps=int(steps*burnin_pct/(1-burnin_pct))
    total_steps=burnin_steps+steps+1

    # init arrays for sample storage
    thetas = np.zeros(total_steps)
    pis = np.zeros((total_steps,K),dtype=int)
    sigmas2 = np.zeros(total_steps)
    taus2 = np.zeros(total_steps)
    betas = np.zeros((total_steps,K))

    # generate initial samples
    thetas[0] = np.random.beta(a=a,b=b,size=1) # fixed initial params a=b=1
    pi0 = np.random.choice(2,size=K,p=[thetas[0],1-thetas[0]])
    pis[0,:] = pi0
    sigmas2[0] = np.random.gamma(shape=a1,scale=1./a2) # fixed initial params a0=a1=0.1
    taus2[0] = stats.invgamma(a=0.5,scale=s**2/2).rvs() # fixed initial param s=0.5
    betas[0,:] = np.sqrt(sigmas2[0]*taus2[0])*np.random.randn(1,K)
    betas[0,:] = np.linalg.inv(x.T@x)@(x.T@y)
    idx = pi0<1
    betas[0,idx]=0

    for t in range(total_steps-1):
        if t<burnin_steps: print('Burn-in: '+str(t+1)+'/'+str(burnin_steps),end='\r')
        if t>=burnin_steps: print('Sampling: '+str(t+1-burnin_steps)+'/'+str(steps),end='\r')

        # summary stats
        sumpi = np.sum(pis[t,:])
        residuals = y-x@betas[t,:]

        # update parameters
        thetas[t+1] = np.random.beta(a+sumpi,b+K-sumpi)
        sigmas2[t+1] = np.random.gamma(shape=a1+N/2,
                                       scale=1/(a2+np.sum(residuals**2)/2))
        taus2[t+1] = stats.invgamma(a=0.5+0.5*sumpi,
                                    scale=s**2/2+0.5*np.sum(betas[t,:]**2)/sigmas2[t]).rvs()

        for k in range(K):
            # loo stats
            tmp_beta = np.copy(betas[t,:])
            tmp_beta[k] = 0. # remove kth param from regression
            z = y-x@tmp_beta
            xk = np.copy(x[:,k])
            cond_var = np.sum(xk**2)+1./taus2[t+1]

            # cat prob
            term1 = np.sum(xk*z)**2 / (2.*sigmas2[t+1]*cond_var)
            l1 = -0.5*np.log(taus2[t+1])+term1-0.5*np.log(cond_var)+np.log(thetas[t+1])
            l2 = np.log(1-thetas[t+1])
            maxl = max(l1,l2)
            ldenominator = maxl+np.log(np.exp(l1-maxl)+np.exp(l2-maxl)) # logsumexp trick
            xi = np.exp(l1-ldenominator)

            # sample cat
            pis[t+1,k] = np.random.choice(2,size=1,p=[1.-xi,xi])
        # end for
        curr_pis = pis[t+1,:]

        ridge_hat = np.linalg.inv((x.T@x)/sigmas2[t+1]+np.eye(K)/(sigmas2[t+1]*taus2[t+1]))
        betas[t+1,:]=np.random.multivariate_normal(mean=ridge_hat@(x.T@y)/sigmas2[t+1],cov=ridge_hat,size=1)
        idx = curr_pis<1
        betas[t+1,idx]=0
    # end for

    return pis[burnin_steps:,:],betas[burnin_steps:,:],thetas[burnin_steps:],sigmas2[burnin_steps:],taus2[burnin_steps:]
