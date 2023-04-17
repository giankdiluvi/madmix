"""
Approximate discrete distributions
with a continuous relaxation (Concrete)
via RealNVP

Implementation in torch
"""

import numpy as np
import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter

"""
########################################
########################################
auxiliary functions
########################################
########################################
"""
def idx_unflatten(x,K2):
    """
    Each x_i is an integer in [0,K1*K2)
    Converts to tuples in [0,K1]x[0,K2]

    Input:
        x  : (d,) array, flattened array
    Output:
        x_ : (2,d) array, unflattened array
    """
    return np.vstack((x//K2,x%K2))

def idx_flatten(x,K2):
    """
    Each x_ij is a tuple in [0,K1]x[0,K2]
    Flattens to integers in [0,K1*K2)

    Input:
        x  : (2,d) array, unflattened array
    Output:
        x_ : (d,) array, flatened array
    """
    return x[0,:]*K2+x[1,:]
#========================================

"""
########################################
########################################
expConcrete distribution
########################################
########################################

taken from
https://pytorch.org/docs/stable/_modules/torch/distributions/relaxed_categorical.html#RelaxedOneHotCategorical
since it is not imported with torch.distributions
"""
from torch.distributions import constraints
from torch.distributions.categorical import Categorical
from torch.distributions.utils import clamp_probs, broadcast_all
from torch.distributions.distribution import Distribution
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import ExpTransform

class ExpRelaxedCategorical(Distribution):
    r"""
    Creates a ExpRelaxedCategorical parameterized by
    :attr:`temperature`, and either :attr:`probs` or :attr:`logits` (but not both).
    Returns the log of a point in the simplex. Based on the interface to
    :class:`OneHotCategorical`.

    Implementation based on [1].

    See also: :func:`torch.distributions.OneHotCategorical`

    Args:
        temperature (Tensor): relaxation temperature
        probs (Tensor): event probabilities
        logits (Tensor): unnormalized log probability for each event

    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables
    (Maddison et al, 2017)

    [2] Categorical Reparametrization with Gumbel-Softmax
    (Jang et al, 2017)
    """
    arg_constraints = {'probs': constraints.simplex,
                       'logits': constraints.real_vector}
    support = constraints.real_vector  # The true support is actually a submanifold of this.
    has_rsample = True

    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        self._categorical = Categorical(probs, logits)
        self.temperature = temperature
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ExpRelaxedCategorical, _instance)
        batch_shape = torch.Size(batch_shape)
        new.temperature = self.temperature
        new._categorical = self._categorical.expand(batch_shape)
        super(ExpRelaxedCategorical, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._categorical._new(*args, **kwargs)

    @property
    def param_shape(self):
        return self._categorical.param_shape

    @property
    def logits(self):
        return self._categorical.logits

    @property
    def probs(self):
        return self._categorical.probs

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        uniforms = clamp_probs(torch.rand(shape, dtype=self.logits.dtype, device=self.logits.device))
        gumbels = -((-(uniforms.log())).log())
        scores = (self.logits + gumbels) / self.temperature
        return scores - scores.logsumexp(dim=-1, keepdim=True)

    def log_prob(self, value):
        K = self._categorical._num_events
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        log_scale = (torch.full_like(self.temperature, float(K)).lgamma() -
                     self.temperature.log().mul(-(K - 1)))
        score = logits - value.mul(self.temperature)
        score = (score - score.logsumexp(dim=-1, keepdim=True)).sum(-1)
        return score + log_scale
#========================================

"""
########################################
########################################
RealNVP architecture
########################################
########################################
"""
class RealNVP(nn.Module):
    """
    Real Non Volume Preserving architecture
    taken from
    https://colab.research.google.com/github/senya-ashukha/real-nvp-pytorch/blob/master/real-nvp-pytorch.ipynb#scrollTo=p5YX9_EW3_EU
    and adapted slightly
    """
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])

    def forward(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def backward(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=-1)
        return z, log_det_J

    def log_prob(self,x):
        z, logp = self.backward(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1)).float()
        logp = self.prior.log_prob(z)
        x = self.forward(z)
        return x

#========================================

def createRealNVP(temp,depth,lprbs,layers=32):
    """
    Wrapper to init a RealNVP class for a problem
    with dimension dim that consists of depth layers

    The reference distribution is a relaxed uniform in exp space

    Inputs:
        temp   : float, temperature of Concrete relaxation
        depth  : int, number of couplings (transformations)
        lprbs  : (dim,) array, target log probabilities
        layers : int, width of the linear layers

    Outputs:
        flow   : Module, RealNVP
    """
    dim=lprbs.shape[0]

    # create channel-wise masks of appropriate size
    masks=torch.zeros((2,dim))
    masks[0,:(dim//2)]=1
    masks[1,(dim-(dim//2)):]=1
    masks=masks.repeat(depth//2,1)

    # define reference distribution
    ref = ExpRelaxedCategorical(torch.tensor([temp]),torch.ones(dim)/dim)

    # define scale and translation architectures
    net_s = lambda: nn.Sequential(
        nn.Linear(dim, layers),
        nn.LeakyReLU(),
        nn.Linear(layers, layers),
        nn.LeakyReLU(),
        nn.Linear(layers, dim),
        nn.Tanh()
    )
    net_t = lambda: nn.Sequential(
        nn.Linear(dim, layers),
        nn.LeakyReLU(),
        nn.Linear(layers, layers),
        nn.LeakyReLU(),
        nn.Linear(layers, dim)
    )
    return RealNVP(net_s, net_t, masks, ref)


"""
########################################
########################################
RealNVP training
########################################
########################################

taken from
https://colab.research.google.com/github/senya-ashukha/real-nvp-pytorch/blob/master/real-nvp-pytorch.ipynb#scrollTo=p5YX9_EW3_EU
and adapted slightly
"""

def trainRealNVP(temp,depth,lprbs,layers=32,max_iters=1000,lr=1e-4,mc_ss=1000,seed=0,verbose=True):
    """
    Train a RealNVP normalizing flow targeting lprbs using the Adam optimizer

    Input:
        temp      : float, temperature of Concrete relaxation
        depth     : int, number of couplings (transformations)
        lprbs     : (dim,) array, target log probabilities
        layers    : int, width of the linear layers
        max_iters : int, max number of Adam iters
        lr        : float, Adam learning rate
        mc_ss     : int, number of samples to draw from target for training
        seed      : int, for reproducinility
        verbose   : boolean, indicating whether to print loss every 100 iterations of Adam
    """
    torch.manual_seed(seed)

    # create flow
    flow=createRealNVP(temp,depth,lprbs,layers=32)
    target = ExpRelaxedCategorical(torch.tensor([temp]),torch.tensor(np.exp(lprbs)))

    # train flow
    sample=target.sample((mc_ss,)).float()
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
