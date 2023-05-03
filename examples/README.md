# Experiments

This directory contains code to reproduce the experiments in the paper.

Generally, each subdirectory contains an experiment 
(or multiple in the case of the toy examples)
in a Jupyter notebook.



## Directory roadmap
- `discrete_toy_examples/` contains three toy discrete examples:
a univariate discrete distribution,
a bivariate discrete distribution,
and a bivariate mixture
- `GMM` has a Gaussian mixture model example
- `ising` includes an Ising model example


## Defining a target log probability

Each Jupyter notebook defines a vectorized log pmf function
for the discrete target distribution.
This function simultaneously returns either the 
joint pmf or the full conditionals (given an axis input).
If you wish to implement your own target log pmf
for an M-dimensional problem where the m-th variable
takes values in {1,...,Km},
the general format for it is this:

```
def lp(x,axis=None):
    # Compute the M-variate log joint and conditional target pmfs
    # The input contains d different M-dimensional points and this fn should be vectorized
    #
    # Inputs:
    #    x    : (M,d) array with state values
    #    axis : int, full conditional to calculate; returns joint if None
    # Outputs:
    #   lprb : if axis is None, (d,) array with log joint; else, (d,Km) array with d conditionals 
    
    if axis==None: 
        # define the log joint and assign it to lprb
        return lprb
    
    # define the axis full conditional and assign it to lprb
    return lprb
```

For specific examples, check the experiments.


## Reproducing the Concrete relaxation results

To reproduce the results of Concrete relaxations,
you have to run a separate `*_run_concrete.py` script
that saves results in  pickle files.
The Jupyter notebooks only load these results.
Due to GitHub size constraints,
the pickle files used in the manuscript could not be committed to this repo,
but are available upon request 
(if you do not want to generate them on your own).
Each experiment requires optimizing 144 Real NVP normalizing flows
with different architecture settings,
which we recommend running in embarrasingly parallel fashion.
Scripts to do this in `.pbs` format are included for each experiment,
along with the settings used in the manuscript.