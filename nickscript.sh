#!/bin/bash
#
#$ -S /bin/bash
#$ -v TST=abc
#$ -M hringbauer@ist.ac.at
#$ -N "MB_METH11"
#$ -m ea
#$ -l mf=4G
#$ -l mem_free=4G
#$ -l h_vmem=4G
#$ -l h_rt=24:00:00
#$ -cwd
#$ -t 1-100:1

export OMP_NUM_THREADS=1
python script_bignick.py $SGE_TASK_ID
# echo $SGE_TASK_ID
