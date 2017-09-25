#!/bin/bash
#
#$ -S /bin/bash
#$ -v TST=abc
#$ -M hringbauer@ist.ac.at
#$ -N "HZALL.v2.5"
#$ -m a    
#$ -l mf=4G
#$ -l mem_free=4G
#$ -l h_vmem=4G
#$ -l h_rt=72:00:00
#$ -cwd
#$ -t 250-500:1

export OMP_NUM_THREADS=1
python script_bignick.py $SGE_TASK_ID
# echo $SGE_TASK_ID
#ea for end/abort
