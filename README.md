# MAD Mix

This is a `python` package that implements [MAD Mix](https://arxiv.org/abs/2308.15613),
a flow-based variational methodology to learn discrete distributions.



## Directory roadmap
Each subdirectory contains README files. Generally:
- `src/` contains the source code for MAD Mix,
including instantiations for discrete-only,
continuous-only (as in [Xu et al. (2022)](https://arxiv.org/abs/2205.07475)),
and mixed discrete and continuous models
- `examples/` has examples where MAD Mix and other methods are used
to learn different distributions, 
some purely discrete and some including continuous variables as well.

## Citing MAD Mix

If you find our code useful, consider citing [our paper](https://arxiv.org/abs/2308.15613).

**BibTeX code for citing MAD Mix**

```
@article{diluvi2023madmix,
  title={Mixed variational flows for discrete variables},
  author={{Gian Carlo} Diluvi and Trevor Campbell and Benjamin Bloem-Reddy},
  journal={arXiv:2308.15613},
  year={2023}
}
```

**APA**

Diluvi, G.C., Bloem-Reddy, B., and Campbell, T. (2023). Mixed variational flows for discrete variables. *arXiv:2308.15613.*
