#!/bin/bash
#
#$ -S /bin/bash
#$ -v TST=abc
#$ -M hringbauer@ist.ac.at
#$ -N "SynthKVar"
#$ -m ea
#$ -l mf=4G
#$ -l mem_free=4G
#$ -l h_vmem=4G
#$ -l h_rt=20:00:00
#$ -cwd
#$ -t 1-11:1

#export OMP_NUM_THREADS=1
Rscript --vanilla single_run.R $SGE_TASK_ID
