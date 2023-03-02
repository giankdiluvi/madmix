import numpy as np
import scipy.stats as stats
import pandas as pd

"""
########################################
########################################
variational approximation functions
########################################
########################################
"""

def lqN(x,u,N,lq0,lp,xi=np.pi/16):
    """
    evaluate variational log likelihood log qN(x,u)
    
    inputs:
        x         : (M,d) array, states of xm
        u         : (M,d) array, values of um
        N         : int, variational parameter; max number of applications of T
        lq0       : function, reference log likelihood (vectorized)
        lp        : function, target log likelihood (vectorized)
        xi        : scalar, uniform shift
        
     outputs:
         logqN(x,u) : (d,) array, likelihood at each datum x_i
    """
    
    if N==1: return lq0(x,u)
    w=np.zeros((N,x.shape[1]))
    w[0,:]=lq0(x,u)
    LJ=np.zeros(x.shape[1])
    for n in range(N-1):
        x,u,tlj=flow(x,u,1,lp,xi,direction='bwd')
        LJ=LJ+tlj
        w[n+1,:]=lq0(x,u)+LJ
    # end for
    return LogSumExp(w)-np.log(N)
        

def randqN(size,N,lp,randq0,xi=np.pi/16):
    """
    generate samples from the variational distribution qN
     
    inputs:
        size   : int, number of samples to generate
        N      : int, variational parameter; max number of applications of T
        lp    – : function, target log likelihood (vectorized)
        randq0 : function, reference distribution sampler
        xi        : scalar, uniform shift
        
     outputs:
       sx      : (M,size) array, x samples from qN
       su      : (M,size,) array, u samples from qN
    """
    
    if N==1: return randq0(size)
    K=np.random.randint(low=0,high=N,size=size)
    x,u=randq0(size)
    for n in range(N-1):
        tx,tu,_ = flow(x[:,K>=n+1],u[:,K>=n+1],steps=1,lp=lp,xi=xi,direction='fwd') # update those with large enough K
        x[:,K>=n+1]=tx
        u[:,K>=n+1]=tu
    # end for
    return x,u
        


"""
########################################
########################################
flow functions
########################################
########################################
"""

def flow(x,u,steps,lp,xi=np.pi/16,direction='fwd'):
    """
    compute T^n(x,u)
    
    inputs:
        x         : (M,d) array, initial states of x
        u         : (M,d) array, initial values of u
        steps     : int, number of applications of T, n
        lp        : function, posterior and conditional pmf
        xi        : scalar, uniform shift
        direction : string, one of 'fwd' (forward map) or 'bwd' (backward map)
        
     outputs:
       x'  : (M,d) array, updated states x'
       u'  : (M,d) array, updated values u'
       ljs : (d,) array, log Jacobians for each sample point
    """
         
    M=x.shape[0]
    ljs=np.zeros(x.shape)
    if steps==0: return x,u,np.zeros(x.shape[1])
    for t in range(steps):
        for m in range(M):
            m_idx = m if direction=='fwd'else M-m-1 # if in reverse, update starting from the end
            tmp_prbs=np.atleast_1d(np.exp(lp(x,axis=m_idx)))
            tmp_prbs=tmp_prbs/np.sum(tmp_prbs,axis=1)[:,np.newaxis]
            tx,tu,tlj=Tm(x[m_idx,:],u[m_idx,:],tmp_prbs,xi,direction=direction)
            x[m_idx,:]=tx
            u[m_idx,:]=tu
            ljs[m_idx,:]=tlj
        # end for
    # end for
    return x,u,np.sum(ljs,axis=0)
            

def Tm(x,u,prbs,xi=np.pi/16,direction='fwd'):
    """
    compute Tm(x,u)
     
    inputs:
        x         : (d,) array, states of xm
        u         : (d,) array, values of um
        prbs      : (d,Km) array, probabilities of Xm|X-m for each of the d xm's
        xi        : scalar, uniform shift
        direction : string, one of 'fwd' (forward map) or 'bwd' (backward map)
        
    outputs:
        xp : (d,) array, updated states xm'
        up : (d,) array, updated values um'
        lj : (d,) array, log Jacobians
    """
            
    if direction=='bwd': xi=-xi # solve modular eqn for inverse by subtracting xi mod 1
    p=getp(x,u,prbs,xi)
    xp=quantile(p,prbs)
    up=(p-cdf(xp-1,prbs))/prbs[np.arange(0,xp.shape[0]),xp]
    return xp,up,np.log(prbs[np.arange(0,x.shape[0]),x])-np.log(prbs[np.arange(0,xp.shape[0]),xp])
    

def getp(x,u,prbs,xi=np.pi/16):
    """
    get proportion from current pair (xm,um)
    equivalent to rho+xi mod 1 in paper
     
    inputs:
       x         : (d,) array, states of xm
       u         : (d,) array, values of um
       prbs      : (d,Km) array, probabilities of Xm|X-m for each of the d xm's
       xi        : scalar, uniform shift
        
    outputs:
       p' : (d,) array, proportion and shifted states p'
    """
    
    p=u*prbs[np.arange(0,x.shape[0]),x]
    F=np.cumsum(prbs,axis=1) # cdf
    p[x>0]=p[x>0]+np.squeeze(F[np.where(x>0),x[x>0]-1]) # vectorized "+prbs[:x] if x>0"
    return (p+xi)%1
    

"""
########################################
########################################
inference
########################################
########################################
"""

def elbo(size,N,lp,lq0,randqN,randq0,xi=np.pi/16):
    """
    estimate ELBO(qN||p)
    
    inputs:
        size   : int, number of samples to generate for MC estimation
        N      : int, variational parameter; max number of applications of T
        lp     : function, target log likelihood (vectorized)
        lq0    : function, reference log likelihood
        randqN : function, variational distribution sampler
        randq0 : function, reference distribution sampler
        xi     : scalar, uniform shift
        
    outputs:
        elbo    : scalar, estimate of the ELBO
    """
            
    tx,tu=randqN(size,N,lp,randq0)
    return np.mean(lp(tx)-lqN(tx,tu,N,lq0,lp,xi))
    

"""
########################################
########################################
auxiliary functions
########################################
########################################
"""

def LogSumExp(w):
    """
    LogSumExp trick
    
    inputs:
        w : (N,d) array, exponents
        
    outputs:
        w' : (N,d) array, log(sum(exp(w)))
    """
    
    wmax = np.amax(w,axis=0)
    return wmax + np.log(np.sum(np.exp(w-wmax[np.newaxis,:]),axis=0))

    
def cdf(x,prbs): 
    """
    cdf of x given prbs (vectorized): F(x)
    
    inputs:
        x    : (d,) array, states of xm
        prbs : (d,Km) array, probabilities of Xm|X-m for each of the d xm's
        
    outputs:
        F(x) : (d,) array, cdf of X at each xi (F(x)_i=F(x_i))
    """
        
    F=np.hstack((np.zeros((prbs.shape[0],1)),np.cumsum(prbs,axis=1))) # adding 0's so F(0)=0
    return F[np.arange(0,x.shape[0]),x+1]
    

def quantile(u,prbs): 
    """
    quantile function of u given prbs (badly vectorized)
    via scipy stats, couldn't implement in native numpy
    
    inputs:
        u    : (d,) array, values of um
        prbs : (d,Km) array, probabilities of Xm|X-m for each of the d xm's
        
    outputs:
        Q(x) : (d,) array, quantile of X at each ui (Q(u)_i=Q(u_i))
    """

    quants=np.zeros(u.shape[0])
    for d in range(u.shape[0]):
        tmprv=stats.rv_discrete(values=(np.arange(0,prbs.shape[1]), prbs[d,:]))
        quants[d]=tmprv.ppf(u[d])
    # end for
    return quants.astype(int)
            

# DEPRECATED - DO NOT USE
def gen_lp(prbs):
    """
    generate an iterable lp function given probs array prbs
    
    inputs:
        prbs : (K1,...,KM) array, probabilities
        
    outputs:
        my_lp : function, obtains joint and conditional probabilities
                my_lp(x)      -> joint at states x
                my_lp(x,axis) -> conditional of x_axis given x_{-axis}
    """
            
    def my_lp(x,axis=None):
        if axis==None: return prbs[tuple(x)] # evaluate lp(x)
        # else return prbs[x_1,x_2,...,x_{m-1},:,x_{m+1},...,x_M] with m=axis
        tmp_prbs=np.ones(prbs.shape[axis]) # init uniform
        tmp_x=np.copy(x)
        for i in range(prbs.shape[axis]):
            tmp_x[axis]=i
            tmp_prbs[i]=prbs[tuple(tmp_x)] 
         # end for
        return tmp_prbs
    return my_lp