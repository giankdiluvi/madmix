#!/bin/bash

#SBATCH --job-name="gmm-penguin-prst"
#SBATCH --account=st-tdjc-1
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16gb
#SBATCH --array=1-36
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gian.diluvi@stat.ubc.ca
#SBATCH --output=/scratch/st-tdjc-1/madmix/gmm/argmax/penguin/output/argmax_%a.txt
#SBATCH --error=/scratch/st-tdjc-1/madmix/gmm/argmax/penguin/errors/argmax_%a.txt
################################################################################

module load python/3.8.10
module load py-argparse/1.4.0
module load py-pip/21.1.2

cd $SLURM_SUBMIT_DIR

export PYTHONPATH=.

cd /arc/project/st-tdjc-1/madmix/examples/gmm/argmax/penguin/

params=`sed "${SLURM_ARRAY_TASK_ID}q;d" gmm_settings_argmax.txt`  # read the settings file in current iteration
param_array=( $params )                            # save current row of settins in param_array

# run code
python gmm_run_argmax_penguin.py --depth ${param_array[1]} --width ${param_array[2]} --lr ${param_array[3]} --max_iters 10001 --idx ${param_array[0]} --outpath '/scratch/st-tdjc-1/madmix/gmm/argmax/penguin/'
