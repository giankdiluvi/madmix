import numpy as np
import scipy.stats as stats
import aux
from ham_mixflows import lap_lm


"""
########################################
########################################
GMM approximation specification
########################################
########################################
"""
def gen_lq0(N,mu0,sigma0):
    """
    Create a log density evaluator q0 in the GMM example

    Inputs:
        N      : int, sample size
        mu0    : (K,D) array, cluster means (K = # of clusters, D=dim of data)
        sigma0 : (K,D,D) array, cluster covariances

    Outputs:
        lq0       : function, reference sampler
    """
    K,D = mu0.shape
    chol = np.linalg.cholesky(sigma0)

    def lq0(xd,ud,xc,rho,uc):
        # Inputs:
        # xd  : (N,B) array, labels
        # ud  : (N,B) array, discrete unifs
        # xc  : (Kp,B) array, continuous vars
        # rho : (Kp,B) array, momenta
        # uc  : (B,) array, continuous unifs
        #
        # Outputs: (B,) array, reference log density
        B=xd.shape[1]

        ws,mus,Hs=madmix_gmm_unflatten(xc,K,D)
        ws=ws/np.sum(ws,axis=0)[None,:]
        Sigmas=HtoSigma(Hs)

        lq  = -N*np.log(K)*np.ones(B) # xd unif ref, ud ref = 0 (uniform[0,1])
        lq += lap_lm(rho) # momenta, uc ref = 0 (uniform[0,1])
        lq += stats.dirichlet(np.ones(K)).logpdf(ws) # weights
        for k in range(K):
            std_mu = np.squeeze(np.matmul(chol[k,:,:],(mus[k,:,:]-mu0[k,:,None]).T[:,:,None]))
            lq += stats.invwishart(df=N/K,scale=sigma0[k,:,:]*N/K).logpdf(Sigmas[k,:,:,:]) # kth cov
            lq += stats.multivariate_normal(mean=np.zeros(D),cov=np.eye(D)).logpdf(std_mu) # kth mean
        # end for

        return lq
    return lq0


def gen_randq0(N,mu0,sigma0,invsigma0):
    """
    Create a sampler for q0 in the GMM example

    Inputs:
        N         : int, sample size
        mu0       : (K,D) array, cluster means (K = # of clusters, D=dim of data)
        sigma0    : (K,D,D) array, cluster covariances
        invsigma0 : (K,D,D) array, cluster inverse covariances (used for Gaussian sampling)

    Outputs:
        randq0    : function, reference sampler
    """
    K,D=mu0.shape

    def randq0(size):
        # Inputs: size : int, sample size

        # discrete vars
        rxd  = np.random.randint(low=0,high=K,size=(N,size))
        rud  = np.random.rand(N,size)

        # continuous vars
        Kp=K+K*D+int(K*D*(1+0.5*(D-1)))
        rrho = np.random.laplace(size=(Kp,size))
        ruc  = np.random.rand(size)

        # weights, means, and covs separately
        rws=np.random.dirichlet(alpha=np.ones(K),size=size).T
        chol=np.linalg.cholesky(sigma0)[:,None,:,:]
        rmus=np.moveaxis(mu0[:,None,:]+np.squeeze(np.matmul(chol,np.random.randn(K,size,D,1))),1,2)
        #rmus=mu0[:,:,None]+np.sum(np.random.randn(K,D,1,size)*invsigma0[:,:,:,None],axis=2)
        rSigmas=np.zeros((K,D,D,size))
        for k in range(K): rSigmas[k,:,:,:]=np.moveaxis(stats.invwishart(N/K,sigma0[k,:,:]*N/K).rvs(size=size),0,2)
        rHs=SigmatoH(rSigmas)
        rxc=madmix_gmm_flatten(rws,rmus,rHs)
        return rxd,rud,rxc,rrho,ruc

    return randq0


"""
########################################
########################################
GMM target specification
########################################
########################################
"""
def gmm_gen_lp(K,y):
    """
    Create a log probability function for the GMM example

    Inputs:
        K : int, number of clusters
        y : (N,D) array, sample (N = # of observations, D=dim of data)

    Outpus:
        lp : function, log pmf of labels
    """
    D=y.shape[1]

    def lp(xd,xc,axis=None):
        # compute the univariate log joint and conditional target pmfs
        #
        # inputs:
        #    xd     : (N,B) array with labels
        #    xc     : (K',B) array with means
        #    axis   : int (0<axis<N), axis to find full conditional; if None then returns the log joint
        # outputs:
        #   ext_lprb : if axis is None, (B,) array with log joint; else, (B,K) array with d conditionals
        N,B=xd.shape

        ws,mus,Hs=madmix_gmm_unflatten(xc,K,D)
        Sigmas=HtoSigma(Hs)

        lprbs=np.zeros((N,K,B))
        for k in range(K):
            for b in range(B):
                lprbs[:,k,b]=stats.multivariate_normal(mus[k,:,b],Sigmas[k,:,:,b]).logpdf(y)
            # end for
        # end for
        lprbs=lprbs-aux.LogSumExp(np.moveaxis(lprbs,1,0))[:,np.newaxis,:]

        ext_lprb=np.zeros((N,B))
        if axis is None:
            ext_lprb=np.zeros((N,B))
            for b in range(B): ext_lprb[:,b]=lprbs[np.arange(0,N),xd[:,b],b]
            return np.sum(ext_lprb,axis=0)
        # end if
        return lprbs[axis,:,:].T
    return lp



def gmm_gen_grad_lp(K,y):
    """
    Create a logp(xc) generator for the GMM example

    Inputs:
        K : int, number of clusters
        y : (N,D) array, sample (N = # of observations, D=dim of data)

    Outpus:
        gen_grad_lp : function, score function generator
    """
    def gen_grad_lp(xd):
        # generate the score function for Hamiltonian dynamics
        #
        # inputs:
        #    xd     : (N,B) array with current labels
        # outputs:
        #   grad_lp : function, vectorized score function ((K',B)->(K',B))
        #
        # Note: K is the number of clusters, D is data dimension,
        # and B is the number of data points (for vectorizing)
        # K'= K (weights) + KxD (means) + KxDxD (covariances)

        idx=(xd[:,None,:]==np.arange(0,K,dtype=int)[None,:,None])               #(N,K,B)
        N_pool=np.sum(idx,axis=0)                                               #(K,B)
        N_pool[N_pool<1]=1                                                      #(K,B) (prevent dividing by 0)
        y_pool=np.sum(y[:,:,None,None]*idx[:,None,:,:],axis=0)/N_pool[None,:,:] #(D,K,B)
        diffs=y[:,:,None,None]-y_pool[None,:,:,:]                               #(N,D,K,B)
        S_pool=np.sum(diffs[:,:,None,:,:]*diffs[:,None,:,:,:],axis=0)           #(D,D,K,B)
        S_pool=S_pool/N_pool[None,None,:,:]                                     #(D,D,K,B)
        S_pool=np.moveaxis(S_pool,2,0)                                          #(K,D,D,B)
        S_poolT=np.transpose(S_pool,axes=(0,2,1,3)) # transpose DxD block, leave first and last axes untouched

        N_,D_,K_,B_= diffs.shape

        def mygrad_lp(xc): # in: (K',B)
            # retrieve unflattened params and invert covariance matrices
            ws,mus,Hs=madmix_gmm_unflatten(xc,K_,D_) #(K,B), (K,D,B),(K,D,D,B)
            Sigmas=HtoSigma(Hs)
            invSigmas=np.zeros((K_,D_,D_,B_))
            for k in range(K_):
                for b in range(B_):
                    invSigmas[k,:,:,b]=np.linalg.inv(Sigmas[k,:,:,b])
                # end for
            # end for
            invSigmasT=np.transpose(invSigmas,axes=(0,2,1,3)) # transpose DxD block, leave first and last axes untouched

            # more quantities
            cluster_diffs=mus-np.moveaxis(y_pool,1,0) #(K,D,B)

            # weight score
            grad_logw=N_pool/ws #(K,B)

            # mean score
            grads_logmu=np.zeros((K_,D_,B_))
            grads_logmu=-N_pool[:,None,:]*np.sum(invSigmas*cluster_diffs[:,None,:,:],axis=1) #(K,D,B)

            # cov score (wild one)
            grads_logsigma=np.zeros((K_,D_,D_,B_))
            grads_logsigma=-0.5*(1+N_pool[:,None,None,:])*invSigmasT #(K,D,D,B)
            grads_logsigma-=0.5*N_pool[:,None,None,:]*cluster_diffs[:,:,None,:]*cluster_diffs[:,None,:,:] #(K,D,D,B)
            tmpinvSigmasT=np.moveaxis(invSigmasT,3,1)
            tmpS_poolT=np.moveaxis(S_poolT,3,1)
            grads_logsigma+=0.5*np.moveaxis(np.matmul(tmpinvSigmasT,np.matmul(tmpS_poolT,tmpinvSigmasT)),1,3) #(K,B,D,D)->(K,D,D,B)

            # calculate jacobian
            jac=HtoSigmaJacobian(Hs) #(K,D**2,D**2,B)
            grads_logsigma=np.matmul(np.moveaxis(jac,3,1),grads_logsigma.reshape(K_,B_,D_**2,1,order='F')) #(K,B,D**2)
            grads_logsigma=grads_logsigma.reshape(K_,D_,D_,B_,order='F') #(K,D,D,B)

            # add derivative wrt determinant jacobian from change of variables
            grads_logsigma+=np.diag(D_-np.arange(D_)+1)[None,:,:,None] #(K,D,D,B)

            return madmix_gmm_flatten(grad_logw,grads_logmu,grads_logsigma) # out: (K',B)
        return mygrad_lp
    return gen_grad_lp



"""
########################################
########################################
GMM auxiliary functions
########################################
########################################
"""

def madmix_gmm_flatten(ws,mus,Hs):
    """
    Flatten weights, meand, and logCholeskys into 2D array

    Inputs:
        ws  : (K,B) array, weights
        mus : (K,D,B) array, cluster means
        Hs  : (K,D,D,B) array, cluster logCholesky matrices

    Outpus:
        xc  : (K',B) array, flattened values

    Note:
    K is the number of clusters, D is data dimension,
    and B is the number of data points (for vectorizing)
    K'= K (weights) + KxD (means) + Kx(D+DChoose2) (covariances)
    """
    K,D,B=mus.shape

    flat_mus=mus.reshape(K*D,B)
    idx=np.tril_indices(D)
    flat_Hs=Hs[:,idx[0],idx[1],:]                     # recover lower triangular entries
    flat_Hs=flat_Hs.reshape(int(K*D*(1+0.5*(D-1))),B) # correct shape
    return np.vstack((ws,flat_mus,flat_Hs))


def madmix_gmm_pack(xd,ud,xc,rho,uc):
    """
    Pack output of MAD Mix into a single np array for pickling

    Inputs:
        xd  : (N,B) array, labels sample (N = # of observations, B = sample size)
        ud  : (N,B) array, discrete unifs sample
        xc  : (K',B) array, continuous variables sample
        rho : (K',B) array, momentum variables sample
        uc  : (B,) array, continuous unifs sample

    Outpus:
        out : (L,B) array, stacked samples

    Note:
    K'= K (weights) + KxD (means) + Kx(D+DChoose2) (covariances)
    L=N+N+K'+K'+1
    """

    return np.vstack((xd,ud,xc,rho,uc[None,:]))


def madmix_gmm_unflatten(xc,K,D):
    """
    Unflatten xc into weights, meand, and covariances

    Inputs:
        xc  : (K',B) array, flattened values

    Outputs:
        ws  : (K,B) array, weights
        mus : (K,D,B) array, cluster means
        Hs  : (K,D,D,B) array, cluster logCholesky matrices

    Note:
    K is the number of clusters, D is data dimension,
    and B is the number of data points (for vectorizing)
    K'= K (weights) + KxD (means) + Kx(D+DChoose2) (covariances)
    """
    B=xc.shape[-1]

    # recover each flattened var
    ws=xc[:K,:]
    flat_mus=xc[K:(K*D+K),:]
    flat_Hs=xc[(K*D+K):,:].reshape(K,int(D*(1+0.5*(D-1))),B)

    # unflatten separately
    mus=flat_mus.reshape(K,D,B)
    Hs=np.zeros((K,D,D,B))
    idx=np.tril_indices(D)
    Hs[:,idx[0],idx[1],:]=flat_Hs

    return ws,mus,Hs

def madmix_gmm_unpack(results,N,K,D):
    """
    Pack output of MAD Mix into a single np array for pickling

    Outputs:
        results : (L,B) array, stacked samples

    Inputs:
        xd  : (N,B) array, labels sample (N = # of observations, B = sample size)
        ud  : (N,B) array, discrete unifs sample
        xc  : (K',B) array, continuous variables sample
        rho : (K',B) array, momentum variables sample
        uc  : (B,) array, continuous unifs sample

    Note:
    K'= K (weights) + KxD (means) + Kx(D+DChoose2) (covariances)
    L=N+N+K'+K'+1
    """
    Kp=K+K*D+int(K*(D+D*(D-1)/2))
    xd=results[:N,:]
    ud=results[N:(2*N),:]
    xc=results[(2*N):(2*N+Kp),:]
    rho=results[(2*N+Kp):(2*N+Kp+Kp),:]
    uc=np.squeeze(results[(2*N+Kp+Kp):,:])

    return xd,ud,xc,rho,uc

def HtoSigma(Hs):
    """
    Transform logCholesky factors into covariance matrices

    Inputs:
        Hs : (K,D,D,B) array, B observations of the K cluster logCholesky factors

    Outpus:
        Sigmas : (K,D,D,B) array, B observations of the K cluster covariances
    """

    idx=np.diag_indices(Hs.shape[1])
    Ls=np.copy(Hs)
    Ls[:,idx[0],idx[1],:]=np.exp(Hs[:,idx[0],idx[1],:])
    Ls=np.moveaxis(Ls,3,1) # so matrices are stacked in last two axes for matmul
    Sigmas=np.matmul(Ls,np.transpose(Ls,axes=(0,1,3,2)))
    return np.moveaxis(Sigmas,1,3)

def SigmatoH(Sigmas):
    """
    Transform covariance matrices into logCholesky factors

    Inputs:
        Sigmas : (K,D,D,B) array, B observations of the K cluster covariances

    Outpus:
        Hs : (K,D,D,B) array, B observations of the K cluster logCholesky factors
    """

    idx=np.diag_indices(Sigmas.shape[1])
    Ls=np.linalg.cholesky(np.moveaxis(Sigmas,3,1))
    Hs=np.copy(Ls)
    Hs[:,:,idx[0],idx[1]]=np.log(Ls[:,:,idx[0],idx[1]])
    return np.moveaxis(Hs,1,3)


#####################
# vec/vech matrices #
#####################

# elimination matrix (4,3)
L2=np.zeros((3,4))
L2[0,0],L2[1,1],L2[2,3]=1.,1.,1.

# duplication matrix (3,4)
D2=np.zeros((4,3))
D2[0,0],D2[1,1],D2[2,1],D2[2,2]=1.,1.,1.,1.

# commutation matrix (D,D)
def comm_mat(D):
    # copied from https://en.wikipedia.org/wiki/Commutation_matrix
    # and modified by Gian Carlo Diluvi
    # determine permutation applied by K
    w = np.arange(D**2).reshape((D, D), order="F").T.ravel(order="F")

    # apply this permutation to the rows (i.e. to each column) of identity matrix and return result
    return np.eye(D**2)[w, :]

def HtoSigmaJacobian(Hs):
    """
    Calculate the Jacobian of the transformation H->Sigma
    Since the transformation is DxD->DxD,
    the Jacobian is a D**2xD**2 matrix
    (to be multiplied with a vectorized matrix vec(grad log p(Sigma)))

    Inputs:
        Hs       : (K,D,D,B) array, cluster logCholesky matrices

    Outputs:
        jacobian : (K,D**2,D**2,B) array, Jacobian matrices
    """
    K,D,B=Hs.shape[0],Hs.shape[1],Hs.shape[3]
    K_mat=comm_mat(D)
    jacobian=np.zeros((K,D**2,D**2,B))

    for k in range(K):
        for b in range(B):
            H=Hs[k,:,:,b]
            L=np.copy(H)
            for d in range(D): L[d,d]=np.exp(H[d,d])

            # get diagexp Jacobian
            expjac=np.zeros((D**2,D**2))+np.eye(D**2)
            for d in np.arange(0,D**2,step=D): expjac[d,d]=L[int(d/4),int(d/4)]

            # get inverse cholesky Jacobian
            LkronI=np.kron(L,np.eye(D))
            choljac=LkronI+np.matmul(LkronI,K_mat)

            # multiply and save result
            jacobian[k,:,:,b]=np.matmul(choljac,expjac)
        # end for
    # end for
    return jacobian


"""
########################################
########################################
Spike and Slab auxiliary functions
########################################
########################################
"""

def sas_unconstrain(theta,tau2,sigma2):
    """
    Map constrained to unconstrained (real) parameters

    Inputs:
        theta  : (B,) array, Categorical probs (in (0,1))
        tau2   : (B,) array, beta variances (>0)
        sigma2 : (B,) array, obs variances (>0)

    Outputs:
        thetau  : (B,) array, unconstrained thetas
        tau2u   : (B,) array, unconstrained tau2s
        sigma2u : (B,) array, unconstrained sigma2s
    """
    thetau  = np.log(theta)-np.log1p(-theta)
    tau2u   = np.log(tau2)
    sigma2u = np.log(sigma2)
    return thetau,tau2u,sigma2u


def sas_constrain(thetau,tau2u,sigma2u):
    """
    Map unconstrained to constrained parameters

    Inputs:
        thetau  : (B,) array, unconstrained thetas
        tau2u   : (B,) array, unconstrained tau2s
        sigma2u : (B,) array, unconstrained sigma2s

    Outputs:
        theta  : (B,) array, Categorical probs (in (0,1))
        tau2   : (B,) array, beta variances (>0)
        sigma2 : (B,) array, obs variances (>0)
    """
    theta  = 1./(1.+np.exp(-thetau))
    tau2   = np.exp(tau2u)
    sigma2 = np.exp(sigma2u)
    return theta,tau2,sigma2


def sas_flatten(theta,tau2,sigma2,beta):
    """
    Flatten parameters into single array
    Parameters can be either constrained or unconstrained

    Inputs:
        theta  : (B,) array, Categorical probs
        tau2   : (B,) array, beta variances
        sigma2 : (B,) array, obs variances
        beta   : (K,B) array, regression coefficients

    Outputs:
        xc     : (3+K,B) array, flattened parameters
    """
    return np.vstack((theta[None,:],tau2[None,:],sigma2[None,:],beta))


def sas_unflatten(xc):
    """
    Unflatten parameters into multiple arrays
    Parameters can be either constrained or unconstrained

    Inputs:
        xc     : (3+K,B) array, flattened parameters

    Outputs:
        theta  : (B,) array, Categorical probs
        tau2   : (B,) array, beta variances
        sigma2 : (B,) array, obs variances
        beta   : (K,B) array, regression coefficients
    """
    theta  = xc[0,:]
    tau2   = xc[1,:]
    sigma2 = xc[2,:]
    beta   = xc[3:,:]
    return theta,tau2,sigma2,beta


def sas_pack(xd,ud,xc,rho,uc):
    """
    Pack output of MAD Mix into a single np array for pickling

    Inputs:
        xd  : (K,B) array, labels sample (N = # of observations, B = sample size)
        ud  : (K,B) array, discrete unifs sample
        xc  : (K',B) array, continuous variables sample
        rho : (K',B) array, momentum variables sample
        uc  : (B,) array, continuous unifs sample

    Outpus:
        out : (L,B) array, stacked samples

    Note:
    K'= 3 (theta, tau2, sigma2) + K (regression coefficients)
    L=K+K+K'+K'+1
    """

    return np.vstack((xd,ud,xc,rho,uc[None,:]))


def sas_unpack(results,K):
    """
    Pack output of MAD Mix into a single np array for pickling

    Outputs:
        results : (L,B) array, stacked samples

    Inputs:
        xd  : (K,B) array, labels sample (N = # of observations, B = sample size)
        ud  : (K,B) array, discrete unifs sample
        xc  : (K',B) array, continuous variables sample
        rho : (K',B) array, momentum variables sample
        uc  : (B,) array, continuous unifs sample

    Note:
    K'= 3 (theta, tau2, sigma2) + K (regression coefficients)
    L=K+K+K'+K'+1
    """
    Kp=3+K
    xd=results[:K,:]
    ud=results[K:(2*K),:]
    xc=results[(2*K):(2*K+Kp),:]
    rho=results[(2*K+Kp):(2*K+Kp+Kp),:]
    uc=np.squeeze(results[(2*K+Kp+Kp):,:])

    return xd,ud,xc,rho,uc


"""
########################################
########################################
Spike and Slab target specification
########################################
########################################
"""

def sas_gen_lp(x,y):
    """
    Create a log probability function for the Spike and Slab example

    Inputs:
        x : (N,K) array, covariates (N = # of observations, K = # of covariates)
        y : (N,) array, response variable observations

    Outpus:
        lp : function, log pmf of labels
    """
    N,K = x.shape

    def lp(xd,xc,axis=None):
        # compute the univariate log joint and conditional target pmfs
        #
        # inputs:
        #    xd     : (K,B) array with regression coeff. indicators (in {0,1})
        #    xc     : (3+K,B) array with real-valued parameters
        #    axis   : int (0<axis<K), axis to find full conditional; if None then returns the log joint
        # outputs:
        #   ext_lprb : if axis is None, (B,) array with log joint; else, (B,2) array with d conditionals
        B=xd.shape[1]

        thetau,tau2u,sigma2u,beta=sas_unflatten(xc) #(B,),(B,),(B,),(K,B)
        theta,tau2,sigma2=sas_constrain(thetau,tau2u,sigma2u)

        lprbs=np.zeros((K,2,B))
        for k in range(K):
            for b in range(B):
                # loo stats
                tmp_beta = np.copy(beta[:,b])
                tmp_beta[k] = 0. # remove kth param from regression
                z = y-x@tmp_beta
                xk = np.copy(x[:,k])
                cond_var = np.sum(xk**2)+1./tau2[b]

                # cat prob
                term1 = np.sum(xk*z)**2 / (2.*sigma2[b]*cond_var)
                l1 = -0.5*np.log(tau2[b])+term1-0.5*np.log(cond_var)+np.log(theta[b])
                l2 = np.log(1-theta[b])
                maxl = max(l1,l2)
                ldenominator = maxl+np.log(np.exp(l1-maxl)+np.exp(l2-maxl)) # logsumexp trick
                lprbs[k,:,b] = np.array([l2,l1])-ldenominator
            # end for
        # end for

        if axis is None:
            ext_lprb=np.zeros((K,B))
            for b in range(B): ext_lprb[:,b]=lprbs[np.arange(0,K),xd[:,b],b]
            return np.sum(ext_lprb,axis=0)
        # end if
        return lprbs[axis,:,:].T
    return lp


def sas_gen_grad_lp(x,y):
    """
    Create a der logp(xc) generator for the GMM example

    Inputs:
        x : (N,K) array, covariates (N = # of observations, K = # of covariates)
        y : (N,) array, response variable observations

    Outpus:
        gen_grad_lp : function, score function generator
    """
    # set hyperparams
    a,b=1.,1.
    a1,a2=0.1,0.1
    s=0.5

    def gen_grad_lp(xd):
        # generate the score function for Hamiltonian dynamics
        #
        # inputs:
        #    xd     : (K,B) array with current labels
        # outputs:
        #   grad_lp : function, vectorized score function ((K',B)->(K',B))
        #
        # Note: K is the number of covariates and
        # B is the number of data points (for vectorizing)
        # K'= 3 (theta, tau2, sigma2) + K (regression coefficients)

        def mygrad_lp(xc): # in: (K',B)
            # retrieve unflattened params
            thetau,tau2u,sigma2u,beta=sas_unflatten(xc) #(B,),(B,),(B,),(K,B)
            theta,tau2,sigma2=sas_constrain(thetau,tau2u,sigma2u)

            # summary stats
            sumpis = np.sum(xd,axis=0) #(B,)
            res = y[None,:]-np.squeeze(np.matmul(x[None,:,:],beta.T[:,:,None])) #(B,N)

            grad_theta   = (a-1.)/theta - (b-1.)/(1.-theta) - 1./(theta*(1.-theta)) #(B,)
            grad_tau2    = s**2/2*tau2**2 - (1.+sumpis)/2*tau2 + np.sum(xd*beta**2,axis=0)/2*sigma2*tau2 #(B,)
            grad_sigma2  = (2.*a2+np.sum(res**2,axis=1))/2*sigma2**2 -(2.*a1+sumpis+y.shape[0])/2.*sigma2 #(B,)
            grad_sigma2 += np.sum(xd*beta**2,axis=0)/2*sigma2*tau2 #(B,)

            grad_beta    = (x.T@y)[None,:] - np.squeeze(np.matmul((x.T@x)[None,:,:],beta.T[:,:,None])) #(B,K)
            grad_beta   += -beta.T/tau2[:,None] #(B,K)

            return sas_flatten(grad_theta,grad_tau2,grad_sigma2,grad_beta.T) # out: (K',B)
        return mygrad_lp
    return gen_grad_lp

"""
###########################################
###########################################
Spike and Slab approximation specification
###########################################
###########################################
"""
def sas_gen_lq0(x,y,nu2):
    """
    Create a log density evaluator q0 in the Spike and Slab example

    Inputs:
        x   : (N,K) array, covariates (N = # of observations, K = # of covariates)
        y   : (N,) array, response variable observations
        nu2 : float, regression coefficients variance

    Outputs:
        lq0 : function, reference sampler
    """
    N,K=x.shape

    # get MLE
    beta_hat = np.linalg.inv(x.T@x)@(x.T@y)

    def lq0(xd,ud,xc,rho,uc):
        # Inputs:
        # xd  : (K,B) array, regression coeff. indicators
        # ud  : (K,B) array, discrete unifs
        # xc  : (3+K,B) array, continuous vars
        # rho : (3+K,B) array, momenta
        # uc  : (B,) array, continuous unifs
        #
        # Outputs: (B,) array, reference log density
        B=xd.shape[1]

        # retrieve unflattened params
        thetau,tau2u,sigma2u,beta=sas_unflatten(xc) #(B,),(B,),(B,),(K,B)
        theta,tau2,sigma2=sas_constrain(thetau,tau2u,sigma2u)

        lq  = -N*np.log(2)*np.ones(B) # xd unif ref, ud ref = 0 (uniform[0,1])
        lq += lap_lm(rho) # momenta; (uc ref = 0 (uniform[0,1]), theta ref = 0 (uniform[0,1]))
        lq += stats.invgamma(a=1.).logpdf(tau2) - np.log(tau2)  # tau2
        lq += stats.gamma(a=1.).logpdf(sigma2) - np.log(sigma2) # sigma2
        lq += stats.multivariate_normal(mean=beta_hat,cov=nu2*np.eye(K)).logpdf(beta.T) # beta

        return lq
    return lq0


def sas_gen_randq0(x,y,nu2):
    """
    Create a sampler for q0 in the Spike and Slab example

    Inputs:
        x   : (N,K) array, covariates (N = # of observations, K = # of covariates)
        y   : (N,) array, response variable observations
        nu2 : float, regression coefficients variance

    Outputs:
        randq0    : function, reference sampler
    """
    N,K=x.shape

    # get MLE
    beta_hat = np.linalg.inv(x.T@x)@(x.T@y)

    def randq0(size):
        # Inputs: size : int, sample size

        # discrete vars
        rxd  = np.random.randint(low=0,high=2,size=(K,size))
        rud  = np.random.rand(K,size)

        # continuous vars
        rrho = np.random.laplace(size=(3+K,size))
        ruc  = np.random.rand(size)

        # relevant continuous params separately
        rtheta = np.random.rand(size)
        rtau2 = 1./np.random.gamma(shape=1., size=size)
        rsigma2 = np.random.gamma(shape=1.,size=size)
        rbeta = beta_hat[:,None]+np.sqrt(nu2)*np.random.randn(K,size)

        # unconstrain and flatten continuous params
        rthetau,rtau2u,rsigma2u=sas_unconstrain(rtheta,rtau2,rsigma2)
        rxc=sas_flatten(rthetau,rtau2u,rsigma2u,rbeta)
        return rxd,rud,rxc,rrho,ruc

    return randq0
