import numpy as np
import scipy.stats as stats
import aux


"""
########################################
########################################
GMM approximation specification
########################################
########################################
"""
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
        # discrete vars
        rxd  = np.random.randint(low=0,high=K,size=(N,size))
        rud  = np.random.rand(N,size)

        # continuous vars
        Kp=K+K*D+int(K*D*(1+0.5*(D-1)))
        rrho = np.random.laplace(size=(Kp,size))
        ruc  = np.random.rand(size)

        # weights, means, and covs separately
        rws=np.random.dirichlet(alpha=np.ones(K),size=size).T
        rmus=mu0[:,:,None]+np.sum(np.random.randn(K,D,1,size)*invsigma0[:,:,:,None],axis=2)
        rSigmas=np.zeros((K,D,D,size))
        for k in range(K): rSigmas[k,:,:,:]=np.moveaxis(stats.invwishart(N,N*sigma0[k,:,:]).rvs(size=size),0,2)
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
            grads_logmu=-N_pool[k,None,:]*np.sum(invSigmas*cluster_diffs[:,None,:,:],axis=1) #(K,D,B)

            # cov score (wild one)
            grads_logsigma=np.zeros((K_,D_,D_,B_))
            grads_logsigma=-0.5*(1+N_pool[k,None,None,:])*invSigmasT #(K,D,D,B)
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
