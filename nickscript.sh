#!/bin/bash
#
#$ -S /bin/bash
#$ -v TST=abc
#$ -M hringbauer@ist.ac.at
#$ -N "AN_BTS"
#$ -m ea
#$ -l mf=8G
#$ -l mem_free=8G
#$ -l h_vmem=8G
#$ -l h_rt=18:00:00
#$ -cwd
#$ -t 1-200:1

export OMP_NUM_THREADS=1
python script_bignick.py $SGE_TASK_ID
