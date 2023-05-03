# Mad Mix source code

This directory contains code to implement Mad Mix for
discrete-only, continuous-only, and mixed models.



## Main code
- `discrete_mixflows.py` contains an instantiation of Mad Mix
for discrete-only models
- `ham_mixflows.py` is a `python` implementation of 
Mixflows via deterministic uncorrected Hamiltonian Monte Carlo,
as introduced in Xu et al. (2022)
- `madmix.py` combines the last two instantiations to produce
flow-based variational approximations for models with 
discrete and continuous variables

## Other scripts
- `aux.py` contains auxiliary code used by all other modules
- `gibbs.py` contains a Gibbs sampler for discrete-only
models that works with the same input as the code in 
`discrete_mixflows.py`. 
It also contains a Gibbs sampler for a Gaussian mixture model
- `concrete.py` contains implementations of Concrete relaxations
and functions to train Real NVP flows for discrete-only models,
as well as code to learn a Gaussian mixture model