# Experiments

This directory contains code to reproduce the experiments in the paper.

Generally, each subdirectory contains an experiment 
(or multiple in the case of the toy examples)
in a Jupyter notebook.



## Directory roadmap
- `summary/` contains the Jupyter notebook to generate the summary Fig. 2 in the paper.
The pickle files with the results of the experiments are not in the repo due to size constraints,
but you can generate them by running each experiment beforehand (which will automatically save these results),
or by opening an issue in the repo requesting the files
- `discrete_toy_examples/` contains the three toy discrete examples:
a 1D discrete distribution,
a 2D discrete distribution,
and a 3D mixture.
It also implements a mixture of MAD Mix flows to learn a discrete mixture
- `GMM/` has two Gaussian mixture model examples,
one with the 
[Palmer penguins](https://github.com/mcnakhaee/palmerpenguins)
data set and another with the 
[waveform](https://hastie.su.domains/ElemStatLearn/datasets/waveform.train)
data set
- `ising/` includes two Ising model examples,
one low-dimensional and another high-dimensional


## Defining a target log probability

Each Jupyter notebook defines a vectorized log pmf function
for the discrete target distribution.
This function returns either the 
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


## Reproducing the dequantization and Concrete relaxation results

To reproduce the results of dequantization and Concrete relaxations,
you have to run a separate `*_run_concrete.py`
or `*_run_dequant.py` script, respectively,
that saves results in  pickle files.
The Jupyter notebooks only load these results.
Due to GitHub size constraints,
the pickle files used in the manuscript could not be committed to this repo,
but are available upon request 
(if you do not want to generate them on your own).
Each experiment requires optimizing 36 (dequantization and argmax flows)
or 144 (Concrete) Real NVP normalizing flows
with different architecture settings,
which we recommend running in parallel fashion.
Scripts to do this in `.pbs` format are included for each experiment,
along with the settings used in the manuscript, in the `sockeye*/` directories.


### Files for each example
Each subdirectory with code for Concrete relaxations and dequantization
has multiple files and subdirectories to run these methods.
Specifically:
- `*.pbs` are files to run the `*_run_concrete.py` script 144 times
or the `*_run_dequant.py` or `*_run_argmax.py` scripts 36 times
with the architecture settings specificied in `*_settings.txt`,
`*_settings_dequant.txt`, or `*_settings_argmax.txt` (respectively)
- `*_settings.txt` contains the settings for the Concrete Real NVP architecture search,
`*_settings_dequant.txt` those for the dequantization Real NVP,
and `*_settings_argmax.txt` those for the argmax flows Real NVP
- `sockeye/` contains cached results and output and error messages from running
the Real NVP architecture search on [UBC ARC Sockeye](https://arc.ubc.ca/ubc-arc-sockeye),
UBC's high-performance computing platform
- The `*.pkl` files with the results from the flows, losses, and cpu times
would be here but are not included in the repo due to GitHub commit size constraints:
we'd be happy to share the files, though&mdash;get in touch!