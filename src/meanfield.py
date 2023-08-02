import numpy as np
import scipy.stats as stats
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

    Outpus:
        lq        : (K,) array, updated variational log probabilities
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


""""
########################
########################
#    mean field gmm    #
########################
########################
"""

# main routine
from scipy.special import digamma as psi
def meanfieldGMM(y,mu0,sigma0,iterations):
    """
    Mean field VI for a Gaussian Mixture Model.
    The algorithm learns q(x,w,mus,Sigmas)
    where x are cluster labels, w are cluster weights,
    mus are cluster means, and Sigmas are cluster covariances.
    The variational approximation decomposes as:
    q(x_n)~Cat(r_{n,1:K}),
    q(w)~Dir(alphas),
    q(Sigma_k)~InvWishart(invW_k,nu_k),
    q(mu_k|Sigma_k)~N(m_k,Sigma_k/beta_k)).

    Inputs:
        y          : (N,D) array, observations (N is no. of obs, d is dimension of each obs)
        mu0        : (K,D) array, initial means (K is number of clusters)
        sigma0     : (K,D,D) array, initial covariances
        iterations : int, number of iterations to run optimization for

    Outputs:
        alphas     : (K,) array, weights Dirichlet approximation variational parameters
        lrs        : (N,K) array, estimated cluster probabilities per observation
        ms         : (K,D) array, means mean variational parameters
        betas      : (K,) array, means covariance variational parameters
        invWs      : (K,D,D) array, covariance invWishart variational parameters
        nus        : (K,) array, covariance invWishart degrees of freedom
    """
    N,D=y.shape
    K=mu0.shape[0]

    # init params
    alpha0=1. # weights w Dirichlet prior param, small = let the data speak
    beta0=0.1*N  # means precision. Default: prior sample size = 10% of observed sample size
    nu0=1.#N-D-2

    # init arrays
    lrs=np.zeros((N,K))-np.log(K) # init at unif
    alphas=np.zeros(K)
    betas=np.zeros(K)
    logLs=np.zeros(K)
    logws=np.zeros(K)
    ms=np.zeros((K,D))
    Ws=np.ones((K,D,D))
    logdetWs=np.zeros(K)
    invWs=np.ones((K,D,D))
    nus=np.ones(K)

    for t in range(iterations):
        print('Iter '+str(t+1)+'/'+str(iterations),end='\r')
        Nks=np.sum(np.exp(lrs),axis=0)


        # variational M
        for k in range(K):
            # get summary statistics
            Nk=Nks[k]
            yk=np.sum(np.exp(lrs[:,k]).reshape(N,1)*y,axis=0)/Nk
            Sk=np.sum(np.exp(lrs[:,k]).reshape(N,1,1)*(y.reshape(N,D,1)-yk.reshape(1,D,1))*(y.reshape(N,1,D)-yk.reshape(1,1,D)),axis=0)/Nk

            # update Dirichlet params
            alphas[k]=alpha0+Nk

            # update mean params
            betas[k]=beta0+Nk
            ms[k,:]=(beta0*mu0[k,:]+Nk*yk)/betas[k]

            # update cov params
            invWs[k,:,:]=sigma0[k,:,:]+Nk*Sk+beta0*Nk/(beta0+Nk)*(yk-mu0[k,:])[:,np.newaxis]*(yk-mu0[k,:])[np.newaxis,:]
            Ws[k,:,:]=np.linalg.inv(invWs[k])
            nus[k]=nu0+Nk

            # define aux quantities
            sign,logdetinvWk = np.linalg.slogdet(invWs[k,:,:])
            logdetWs[k]=-logdetinvWk
            logLs[k]=np.sum(psi(0.5*(nus[k]+1-np.arange(1,D+1))))+D*np.log(2)+logdetWs[k]
            logws[k]=psi(alphas[k])+psi(np.sum(alphas))
            # end for

        # variational E
        for k in range(K):
            lrs[:,k]=logws[k]+0.5*logLs[k]-0.5*D/betas[k]
            gauss_term=0.5*nus[k]*np.sum(np.sum((y.reshape(N,D,1)-ms[k,:].reshape(1,D,1))*Ws[k]*(y.reshape(N,1,D)-ms[k,:].reshape(1,1,D)),axis=-1),axis=-1)
            lrs[:,k]-=gauss_term
        # end for

        # normalize weights
        lrs=lrs-LogSumExp(lrs.T)[:,np.newaxis]
    # end for

    return alphas,lrs,ms,betas,invWs,nus



##################
##################
# auxiliary fns  #
##################
##################

def meanfield_gmm_elbo(B,lp,alphas,lrs,ms,betas,invWs,nus,y,mu0,Sigma0):
    """
    Estimate ELBO of approximation

    Inputs:
        B       : int, sample size for Monte Carlo estimation
        lp      : function, posterior log density (unnormalized)
        alphas  : (K,) array, Dirichlet params
        lrs     : (N,K) array, labels params
        ms      : (K,D) array, cluster means location params
        betas   : (K,) array, cluster means spread params
        invWs   : (K,D,D) array, cluster covariances scale params
        nus     : (K,) array, cluster covariances df params
        y       : (N,D) array, observations
        mu0     : (K,D,B) array, prior cluster means
        Sigma0  : (K,D,D,B) array, prior cluster covariances

    Outputs:
        e  : float, estimate of the ELBO
    """
    rxd,rw,rmus,rSigmas = meanfield_gmm_rq(B,alphas,lrs,ms,betas,invWs,nus)
    llq = meanfield_gmm_lq(rxd,rw,rmus,rSigmas,alphas,lrs,ms,betas,invWs,nus)
    llp = lp(rxd,rw,rmus,rSigmas,y,mu0,Sigma0)
    return -np.mean(llq-llp)


def meanfield_gmm_lq(xd,w,mus,Sigmas,alphas,lrs,ms,betas,invWs,nus):
    """
    Compute the log density of the mean field approximation to a GMM

    Inputs:
        xd     : (N,B) array, labels to evaluate log pmf
        w      : (K,B) array, weights to evaluate log pdf
        mus    : (K,D,B) array, means to evaluate log pdf
        Sigmas : (K,D,D,B) array, covariances to evaluate log pdf
        alphas : (K,) array, Dirichlet params
        lrs    : (N,K) array, labels params
        ms     : (K,D) array, cluster means location params
        betas  : (K,) array, cluster means spread params
        invWs  : (K,D,D) array, cluster covariances scale params
        nus    : (K,) array, cluster covariances df params

    Outputs:
        lq     : (B,) array, log densities
    """
    N,B  = xd.shape
    K,D,_= mus.shape
    chol = np.linalg.cholesky(np.moveaxis(Sigmas,3,1))  #(K,B,D,D)

    lq = stats.dirichlet(alphas).logpdf(w) # weights
    for k in range(K):
        std_mu = np.squeeze(np.matmul(chol[k,:,:,:],(mus[k,:,:]-ms[k,:,None]).T[:,:,None]))
        lq += stats.multivariate_normal(mean=np.zeros(D), cov=np.eye(D)).logpdf(std_mu) # kth mean
        lq += stats.invwishart(df=nus[k],scale=invWs[k,:,:]).logpdf(Sigmas[k,:,:,:])    # kth covariance
    # end for
    for n in range(N):
        lrsn=lrs[n,:]
        lq += lrsn[xd[n,:].astype(int)] # labels
    # end for
    return lq


def meanfield_gmm_rq(size,alphas,lrs,ms,betas,invWs,nus):
    """
    Compute the log density of the mean field approximation to a GMM

    Inputs:
        size    : int, sample size
        alphas  : (K,) array, Dirichlet params
        lrs     : (N,K) array, labels params
        ms      : (K,D) array, cluster means location params
        betas   : (K,) array, cluster means spread params
        invWs   : (K,D,D) array, cluster covariances scale params
        nus     : (K,) array, cluster covariances df params

    Outputs:
        rxd     : (N,size) array, sample labels
        rw      : (K,size) array, sample weights
        rmus    : (K,D,size) array, sample cluster means
        rSigmas : (K,D,D,size) array, sample cluster covariances
    """
    N,K = lrs.shape
    _,D = ms.shape

    # labels via Gumbel-max trick
    G = np.random.gumbel(size=(N,K,size))
    rxd = np.argmax(lrs[:,:,None]+G,axis=1)

    # continuous vars
    rw = stats.dirichlet(alphas).rvs(size).T # weights
    rSigmas = np.zeros((K,D,D,size))
    for k in range(K): rSigmas[k,...]=np.moveaxis(stats.invwishart(df=nus[k],scale=invWs[k,:,:]).rvs(size),0,2) # covariances
    chol = np.linalg.cholesky(np.moveaxis(rSigmas,3,1)) #(K,B,D,D)
    rmus = ms[...,None]+np.moveaxis(np.squeeze(np.matmul(chol,np.random.randn(K,size,D,1))),1,2) # means

    return rxd,rw,rmus,rSigmas
