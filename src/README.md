# MAD Mix source code

This directory contains code to implement MAD Mix for
discrete-only, continuous-only, and mixed models.
It also contains implementations of other algorithms
used for comparison in the paper.
The `examples/` directory uses these scripts to approximate
multiple different targets.


## Main code
- `discrete_mixflows.py` contains an instantiation of MAD Mix
for discrete-only models
- `ham_mixflows.py` is a `python` implementation of 
Mixflows via deterministic uncorrected Hamiltonian Monte Carlo,
as introduced in [Xu et al. (2022)](https://arxiv.org/abs/2205.07475)
- `madmix.py` combines the last two instantiations to produce
flow-based variational approximations for models with 
discrete and continuous variables

## Auxiliary scripts
- `madmix_aux.py` has auxiliary code necessary to fit a GMM with MAD Mix
(e.g., score functions and log-Cholesky decomposition helpers)
- `aux.py` contains auxiliary code (LogSumExp, pickle cacher/loader) 
used by all other modules

## Other algorithms
- `gibbs.py` contains a Gibbs sampler for discrete-only
models that works with the same input as the code in 
`discrete_mixflows.py` 
- `meanfield.py` implements mean-field VI for discrete models up to 3D,
Ising models, and GMMs
It also contains a Gibbs sampler for a Gaussian mixture model
- `concrete.py` contains implementations of Concrete relaxations
and functions to train Real NVP flows for discrete-only models,
as well as code to learn a Gaussian mixture model
and to do sparse Bayesian regression with a Spike-and-Slab prior
- `dequantization.py` contains an implementation of uniform dequantization
and functions to train Real NVP flows for discrete-only models,
as well as code to learn a Gaussian mixture model
and to do sparse Bayesian regression with a Spike-and-Slab prior
- `argmax_flows.py` contains an implementation of argmax flows
and functions to train Real NVP flows for discrete-only models,
as well as code to learn a Gaussian mixture model
and to do sparse Bayesian regression with a Spike-and-Slab prior