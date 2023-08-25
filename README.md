# MAD Mix

This is a `python` package that implements MAD Mix,
a flow-based variational methodology to learn discrete distributions.



## Directory roadmap
Each subdirectory contains README files. Generally:
- `src/` contains the source code for MAD Mix,
including instantiations for discrete-only,
continuous-only (as in [Xu et al. (2022)](https://arxiv.org/abs/2205.07475)),
and mixed discrete and continuous models
- `examples/` has examples where MAD Mix and other methods are used
to learn different distributions, 
some purely discrete and some including continuous variables as well
