
#SBATCH --ntasks=1
#SBATCH --job-name=coalescence_sims
#SBATCH --time=24:00:00
#SBATCH --mem=1G
#SBATCH --mail-user=hringbauer@ist.ac.at
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --array=0-2
unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=1


python coalescence_sims.py $SLURM_ARRAY_TASK_ID

