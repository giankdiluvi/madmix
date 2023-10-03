# Spike-and-Slab model

The `SpikeAndSlab.ipynb` Jupyter notebook contains code
to reproduce the Spike-and-Slab examples.
You should be able to run the whole notebook to get the results,
except those of dequantization and the Concrete distribution.
Check the `examples/`` README to read more about that.



## Directory roadmap
- `SpikeAndSlab.ipynb` contains both GMM examples
- `sockeye/` contains cached results and output and error messages from running
the Real NVP architecture search on [UBC ARC Sockeye](https://arc.ubc.ca/ubc-arc-sockeye),
UBC's high-performance computing platform
- `sockeye_run/` includes the files necessary to learn the Real NVP Concrete relaxation
- `meanfield/` has the `R` code to fit a mean-field VI algorithm 
to both data sets using the 
[sparsevb](https://cran.r-project.org/web/packages/sparsevb/sparsevb.pdf).
package
- `dat/` includes both data sets.
Specifically, `prst_dat_x.csv` and `prst_dat_y.csv`
contain the covariates and response variable, respectively,
for the prostate cancer data set
and `spr_dat_x.csv` and `spr_dat_y.csv` the respective data sets
for the superconductivity data sets.
These files are generated in the `SpikeAndSlab.ipynb` file,
which imports the prostate cancer data set from a url
and the reads the superconductivity data set from `train.csv`
- `results/` subdirectory contains
the cpu times from the meanfield algorithm.
It should also contain cached pickle files from MAD Mix,
but due to GitHub commit size constraints it cannot be uploaded.
You can still run the MAD Mix code in a reasonable amount of time
by setting `RUN=True` in the appropriate cells.


For more info on the files in `sockeye` and `sockeye_run`,
check the README of the `examples/` directory.