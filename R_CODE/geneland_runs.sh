#!/bin/bash
#
#$ -S /bin/bash
#$ -v TST=abc
#$ -M hringbauer@ist.ac.at
#$ -N "2400i200l"
#$ -m ea
#$ -l mf=2G
#$ -l mem_free=2G
#$ -l h_vmem=2G
#$ -l h_rt=100:00:00
#$ -cwd
#$ -t 1-10:1

export OMP_NUM_THREADS=1
Rscript --vanilla single_run.R $SGE_TASK_ID