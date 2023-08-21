import numpy as np
import sys
import concrete

import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter



def create_dequant_RealNVP(dim,depth,width):
    """
    Wrapper to init a RealNVP flow for a dequantization problem
    with dimension dim that consists of depth layers

    The reference distribution is an isotropic Gaussian

    Inputs:
        dim   : int, dimension of data
        depth : int, number of couplings (transformations)
        width : int, width of the linear layers

    Outputs:
        flow   : Module, RealNVP
    """

    # create channel-wise masks of appropriate size
    masks=torch.zeros((2,dim))
    masks[0,:(dim//2)]=1
    masks[1,(dim-(dim//2)):]=1
    masks=masks.repeat(depth//2,1)

    # define reference distribution
    ref = torch.distributions.MultivariateNormal(torch.zeros(dim), torch.eye(dim))

    # define scale and translation architectures
    net_s = lambda: nn.Sequential(
        nn.Linear(dim, width),
        nn.LeakyReLU(),
        nn.Linear(width, width),
        nn.LeakyReLU(),
        nn.Linear(width, dim),
        nn.Tanh()
    )
    net_t = lambda: nn.Sequential(
        nn.Linear(dim, width),
        nn.LeakyReLU(),
        nn.Linear(width, width),
        nn.LeakyReLU(),
        nn.Linear(width, dim)
    )
    return concrete.RealNVP(net_s, net_t, masks, ref)



def train_dequant_discrete(depth,sample,width=32,max_iters=1000,lr=1e-4,seed=0,verbose=True):
    """
    Train a dequantized RealNVP normalizing flow using the Adam optimizer

    Input:
        depth     : int, number of couplings (transformations)
        sample    : (dim,B) array, quantized sample from target (e.g., from Gibbs sampler)
        width     : int, width of the linear layers
        max_iters : int, max number of Adam iters
        lr        : float, Adam learning rate

        seed      : int, for reproducibility
        verbose   : boolean, indicating whether to print loss every 100 iterations of Adam
    """
    torch.manual_seed(seed)

    # create flow
    dim,B = sample.shape
    flow = create_dequant_RealNVP(dim,depth,width)

    # dequantize sample
    training_sample = torch.from_numpy(sample.T) + torch.rand(B,dim)

    # train flow
    optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=lr)
    losses=np.zeros(max_iters)

    for t in range(max_iters):
        loss = -flow.log_prob(training_sample).mean()
        losses[t]=loss

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if verbose and t%(max_iters//10) == 0: print('iter %s:' % t, 'loss = %.3f' % loss)
    # end for
    return flow,losses




"""
########################################
########################################
GMM functions
########################################
########################################
"""


from torch.distributions.distribution import Distribution
class GMMRef_dequant(Distribution):
    """
    Reference distribution for labels, weights, means, and covariances of a GMM.
    Gaussian for the (dequantized) labels,
    Dirichlet for the weights,
    Gaussians for the means,
    and InverseWishart for the covariances

    Inputs:
        N    : int, number of observations from the GMM
        K    : int, mixture size
        tau0 : float, labels and means prior precision
    """
    def __init__(self, N,K,D,tau0=1.):
        self.dirichlet = torch.distributions.dirichlet.Dirichlet(concentration=torch.ones(K))
        self.gauss  = torch.distributions.MultivariateNormal(torch.zeros(D), torch.eye(D)/np.sqrt(tau0))
        self.std_gauss = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.invwis = torch.distributions.wishart.Wishart(df=N/K,covariance_matrix=torch.eye(D))
        self.K = K
        self.N = N
        self.D = D

    def log_prob(self, value):
        relcat_lp = torch.zeros(value.shape[0])
        for n in range(self.N): relcat_lp += self.std_gauss.log_prob(value[...,n])

        idx=np.diag_indices(self.D)
        xc=value[...,self.N:].T
        ws,mus,Hs=concrete.concrete_gmm_unflatten(xc,self.K,self.D) #(K,B),(K,D,B),(K,D,D,B)
        Sigmas=concrete.HtoSigma(Hs) #(K,D,D,B)
        xc_lp = torch.zeros(ws.shape[1])
        for k in range(self.K):
            xc_lp += self.invwis.log_prob(torch.moveaxis(Sigmas[k,...],2,0))
            xc_lp += self.D*torch.log(torch.tensor([2])) + torch.sum((self.D-torch.arange(self.D)+1)[:,None]*Hs[k,idx[0],idx[1],:],axis=0) # determinant Jacobian of log-Cholesky decomposition
            xc_lp += self.gauss.log_prob(mus[k,...].T)
        # end for
        return relcat_lp+xc_lp

    def sample(self,sample_shape=torch.Size()):
        labels_sample=self.std_gauss.sample(sample_shape)
        for n in range(self.N-1): labels_sample = torch.hstack((labels_sample,self.std_gauss.sample(sample_shape)))

        ws_sample = self.dirichlet.sample(sample_shape).T
        sigmas_sample = torch.zeros((self.K,self.D,self.D,sample_shape[0]))
        mus_sample = torch.zeros((self.K,self.D,sample_shape[0]))
        for k in range(self.K):
            sigmas_sample[k,...] = torch.moveaxis(self.invwis.sample(sample_shape),0,2)
            mus_sample[k,...] = self.gauss.sample(sample_shape).T
        # end for
        Hs_sample = concrete.SigmatoH(sigmas_sample)
        xc_sample = concrete.concrete_gmm_flatten(ws_sample,mus_sample,Hs_sample).T
        return torch.hstack((labels_sample,xc_sample))
#========================================


def create_dequant_gmm_RealNVP(depth,width,N,K,D,tau0):
    """
    Wrapper to init a RealNVP flow for a dequantization problem
    with dimension dim that consists of depth layers

    The reference distribution is an isotropic Gaussian

    Inputs:
        depth  : int, number of couplings (transformations)
        width  : int, width of the linear layers
        N      : int, number of observations from the GMM
        K      : int, mixture size
        D      : int, dimension of data
        tau0   : float, prior precision for means

    Outputs:
        flow   : Module, RealNVP
    """
    dim=int(N+K+K*D+K*(D+D*(D-1)/2))

    # create channel-wise masks of appropriate size
    masks=torch.zeros((2,dim))
    masks[0,:(dim//2)]=1
    masks[1,(dim-(dim//2)):]=1
    masks=masks.repeat(depth//2,1)

    # define reference distribution
    ref = GMMRef_dequant(N,K,D,tau0)

    # define scale and translation architectures
    net_s = lambda: nn.Sequential(
        nn.Linear(dim, width),
        nn.LeakyReLU(),
        nn.Linear(width, width),
        nn.LeakyReLU(),
        nn.Linear(width, dim),
        nn.Tanh()
    )
    net_t = lambda: nn.Sequential(
        nn.Linear(dim, width),
        nn.LeakyReLU(),
        nn.Linear(width, width),
        nn.LeakyReLU(),
        nn.Linear(width, dim)
    )
    return concrete.RealNVP(net_s, net_t, masks, ref, gmm=True)


def gmm_dequant_sample(pred_x,pred_w,pred_mus,pred_sigmas):
    """
    Generate sample for learning GMM with RealNVP
    dequantizing the labels

    Inputs:
        pred_x      : (steps,N) array, predicted label values (from `gibbs.gibbs_gmm`)
        pred_w      : (steps,K) array, predicted weights (from `gibbs.gibbs_gmm`)
        pred_mus    : (steps,K,D) array, predicted means (from `gibbs.gibbs_gmm`)
        pred_sigmas : (steps,K,D,D) array, predicted covariance matrices (from `gibbs.gibbs_gmm`)

    Outputs:
        dequant_sample : (steps,K'') array, samples to be used in training

    Note: K'' = N (labels) + K (weights) + KD (means) + Kx(D+DChoose2) (covariances, log-Cholesky decomposed)
    """
    # convert gibbs output to torch tensors
    xs     = torch.from_numpy(pred_x)
    ws     = torch.from_numpy(pred_w)
    mus    = torch.from_numpy(pred_mus)
    sigmas = torch.from_numpy(pred_sigmas)

    # dequantize label sample
    xd = xs + torch.rand(pred_x.shape)

    # deal with continuous variables
    Hs=concrete.SigmatoH(torch.moveaxis(sigmas,0,3))
    xc=concrete.concrete_gmm_flatten(ws.T,torch.moveaxis(mus,0,2),Hs).T

    # merge everything
    dequant_sample=torch.hstack((xd,xc)).float()

    return dequant_sample


def train_dequant_gmm(depth,N,K,D,tau0,sample,width=32,max_iters=1000,lr=1e-4,seed=0,verbose=True):
    """
    Train a RealNVP normalizing flow targeting a GMM with dequantized cluster labels

    Input:
        depth     : int, number of couplings (transformations)
        N         : int, number of observations from the GMM
        K         : int, mixture size
        D         : int, dimension of data
        tau0      : float, prior precision for means
        sample    : (B,K'') array, samples from target for training; B is the Monte Carlo sample size
        width     : int, width of the linear layers
        max_iters : int, max number of Adam iters
        lr        : float, Adam learning rate
        seed      : int, for reproducinility
        verbose   : boolean, indicating whether to print loss every 100 iterations of Adam

    Output:
        flow      : distribution, trained normalizing flow
        losses    : (maxiters,) array with loss traceplot

    Note: K'' = N (labels) + K (weights) + KD (means) + Kx(D+DChoose2) (covariances, which will be log-Cholesky decomposed)
    """
    torch.manual_seed(seed)

    # create flow
    flow=create_dequant_gmm_RealNVP(depth,width,N,K,D,tau0)

    # train flow
    optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=lr)
    losses=np.zeros(max_iters)

    for t in range(max_iters):
        loss = -flow.log_prob(sample).mean()
        losses[t]=loss

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if verbose and t%(max_iters//10) == 0: print('iter %s:' % t, 'loss = %.3f' % loss)
    # end for
    return flow,losses



def dequant_gmm_unpack(x,N,K,D):
    """
    Unpack x=[xd,xc] into label normalized logprbs, weights, means, and covariances

    Inputs:
        x      : (B,K'') array, flattened values

    Outputs:
        xd     : (N,B) array, label log probabilities
        ws     : (K,B) array, weights
        mus    : (K,D,B) array, cluster means
        Sigmas : (K,D,D,B) array, cluster covariance matrices

    Note:
    K is the number of clusters, D is data dimension,
    and B is the number of data points (for vectorizing)
    K'' = N (labels) + K (weights) + KxD (means) + Kx(D+DChoose2) (covariances)

    Note:
    The label probabilities tensor will be converted to a np array and gradients will be detached
    """
    x=x.T
    B=x.shape[1]

    xd=x[:N,:].reshape(N,B)
    xd=xd.detach().numpy()

    xc=x[N:,:]
    ws,mus,Hs=concrete.concrete_gmm_unflatten(xc,K,D)
    Sigmas=concrete.HtoSigma(Hs)

    return xd,ws,mus,Sigmas
#========================================
