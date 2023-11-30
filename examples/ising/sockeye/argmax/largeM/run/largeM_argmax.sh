#!/bin/bash

#SBATCH --job-name="ising-largeM-argmax"
#SBATCH --account=st-tdjc-1
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16gb
#SBATCH --array=1-36
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gian.diluvi@stat.ubc.ca
#SBATCH --output=/scratch/st-tdjc-1/madmix/ising/argmax/largeM/output/argmax_%a.txt
#SBATCH --error=/scratch/st-tdjc-1/madmix/ising/argmax/largeM/errors/argmax_%a.txt
################################################################################

module load python/3.8.10
module load py-argparse/1.4.0
module load py-pip/21.1.2

cd $SLURM_SUBMIT_DIR

export PYTHONPATH=.

cd /scratch/st-tdjc-1/madmix/ising/argmax/largeM/run

params=`sed "${SLURM_ARRAY_TASK_ID}q;d" ising_settings_argmax.txt`  # read the settings file in current iteration
param_array=( $params )                            # save current row of settins in param_array

# run code
python ising_run_argmax.py -M 'large' --depth ${param_array[1]} --width ${param_array[2]} --lr ${param_array[3]} --max_iters 10001 --idx ${param_array[0]} --outpath '/scratch/st-tdjc-1/madmix/ising/argmax/largeM/'
