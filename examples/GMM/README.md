# Gaussian mixture model

The `Gaussian_mixture_model.ipynb` Jupyter notebook contains code
to reproduce the GMM examples.
You should be able to run the whole notebook to get the results,
except those of dequantization and the Concrete distribution.
Check the `examples/`` README to read more about that.



## Directory roadmap
- `GaussianMixtureModel.ipynb` contains both GMM examples
- `sockeye/` contains cached results and output and error messages from running
the Real NVP architecture search on [UBC ARC Sockeye](https://arc.ubc.ca/ubc-arc-sockeye),
UBC's high-performance computing platform
- `sockeye_run/` includes the files necessary to learn the Real NVP Concrete relaxation
- A `results/` subdirectory containing
cached pickle files from MAD Mix should also be included here
but due to GitHub commit size constraints it cannot be uploaded.
You can still run the MAD Mix code in a reasonable amount of time
by setting `RUN=True` in the appropriate cells.


For more info on the files in `sockeye` and `sockeye_run`,
check the README of the `examples/` directory.