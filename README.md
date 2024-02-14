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
@inproceedings{diluvi2024madmix,
  title={Mixed variational flows for discrete variables},
  author={{Gian Carlo} Diluvi and Benjamin Bloem-Reddy and Trevor Campbell},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  year={2024}
  }   
```

**APA**

Diluvi, G.C., Bloem-Reddy, B., and Campbell, T. 
Mixed variational flows for discrete variables. 
In *International Conference on Artificial Intelligence and Statistics*, 2024.
