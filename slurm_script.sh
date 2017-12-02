#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name="BITCOIN_MINING (Just Joking)"
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --mail-user=hringbauer@ist.ac.at
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --array=0-1499
unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=1


python script_bignick.py $SLURM_ARRAY_TASK_ID

