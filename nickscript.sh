#!/bin/bash
#
#$ -S /bin/bash
#$ -v TST=abc
#$ -M hringbauer@ist.ac.at
#$ -N "MultiIndM2"
#$ -m ea
#$ -l mf=20G
#$ -l mem_free=20G
#$ -l h_vmem=20G
#$ -l h_rt=24:00:00
#$ -cwd
#$ -t 1-100:1

export OMP_NUM_THREADS=1
python script_bignick.py $SGE_TASK_ID
# echo $SGE_TASK_ID
