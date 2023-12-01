import numpy as np
import sys
import concrete
import dequantization

import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter



"""
########################################
########################################
training
########################################
########################################
"""


def create_argmax_RealNVP(dim,depth,width):
    """
    Wrapper to init a RealNVP flow for an argmax problem
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



def train_argmax_discrete(depth,sample,width=32,max_iters=1000,lr=1e-4,seed=0,verbose=True):
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
    B,dim = sample.shape # dim = M*max, with max the largest value taken by x in sample
    flow = create_argmax_RealNVP(dim,depth,width)

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



"""
########################################
########################################
sample generation
########################################
########################################
"""

def argmax_discsample_gen(sample):
    """
    Transform a sample from a discrete variable for argmax training

    Input:
        sample: (M,B) array, initial sample of M discrete variables and sample size B

    Output:
        out   : (B,M*max) array, sample for training an argmax flow,
                                 with max the largest value taken by the samples

    Note: np.argmax(out,axis=1)==sample at each m
    """
    M,B = sample.shape
    maxs = 1+np.amax(sample)
    out = np.zeros((M*maxs,B))

    for m in range(M):
        gauss = np.random.randn(maxs,B)
        T = gauss[sample[m,:],np.arange(B)] # get entries where max happens as per sample
        tmp_out = T[None,:]-np.log1p(np.exp(T[None,:]-gauss)) # threshold
        tmp_out[sample[m,:],np.arange(B)]=T # make sure that max happens at T
        out[maxs*m+np.arange(maxs),:]=tmp_out # add to sample
    # end for

    return torch.from_numpy(out.T).float()


def argmax_gmmsample_gen(pred_x,pred_w,pred_mus,pred_sigmas):
    """
    Generate sample for learning GMM with RealNVP
    through argmax flows for the labels

    Inputs:
        pred_x      : (steps,N) array, predicted label values (from `gibbs.gibbs_gmm`)
        pred_w      : (steps,K) array, predicted weights (from `gibbs.gibbs_gmm`)
        pred_mus    : (steps,K,D) array, predicted means (from `gibbs.gibbs_gmm`)
        pred_sigmas : (steps,K,D,D) array, predicted covariance matrices (from `gibbs.gibbs_gmm`)

    Outputs:
        dequant_sample : (steps,K'') array, samples to be used in training

    Note: K'' = NK (labels) + K (weights) + KD (means) + Kx(D+DChoose2) (covariances, log-Cholesky decomposed)
    """
    # convert gibbs output to torch tensors
    #xs     = torch.from_numpy(pred_x)
    ws     = torch.from_numpy(pred_w)
    mus    = torch.from_numpy(pred_mus)
    sigmas = torch.from_numpy(pred_sigmas)

    # dequantize label sample
    xd = argmax_discsample_gen(pred_x.T)

    # deal with continuous variables
    Hs=concrete.SigmatoH(torch.moveaxis(sigmas,0,3))
    xc=concrete.concrete_gmm_flatten(ws.T,torch.moveaxis(mus,0,2),Hs).T

    # merge everything
    argmax_sample=torch.hstack((xd,xc)).float()

    return argmax_sample



def argmax_sassample_gen(pred_pi,pred_beta,pred_theta,pred_sigma2,pred_tau2):
    """
    Generate sample for learning Spike-and-Slab with RealNVP
    through argmax flows for the inclusion idx

    Inputs:
        pred_pi     : (steps,K) array, predicted inclusion idx (from `gibbs.gibbs_gmm`)
        pred_beta   : (steps,K) array, predicted regression coefficients (from `gibbs.gibbs_gmm`)
        pred_theta  : (steps,) array, predicted inclusion prbs (from `gibbs.gibbs_gmm`)
        pred_sigma2 : (steps,) array, predicted observation vars (from `gibbs.gibbs_gmm`)
        pred_tau2   : (steps,) array, predicted beta vars (from `gibbs.gibbs_gmm`)

    Outputs:
        dequant_sample : (steps,K') array, samples to be used in training

    Note: K' = 2K (pi) + 3 (theta, sigma2, tau2) + K (beta)
    """
    # convert gibbs output to torch tensors
    betau  = torch.from_numpy(pred_beta.T)
    theta  = torch.from_numpy(pred_theta)
    sigma2 = torch.from_numpy(pred_sigma2)
    tau2   = torch.from_numpy(pred_tau2)

    # unconstrain results
    thetau,tau2u,sigma2u = dequantization.sas_unconstrain_torch(theta,tau2,sigma2)

    # dequantize label sample
    xd = argmax_discsample_gen(pred_pi.T).T

    # deal with continuous variables
    xc = dequantization.sas_flatten_torch(thetau,tau2u,sigma2u,betau)

    # merge everything
    argmax_sample=torch.vstack((xd,xc)).float().T

    return argmax_sample
