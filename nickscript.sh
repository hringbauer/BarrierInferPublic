#!/bin/bash
#
#$ -S /bin/bash
#$ -v TST=abc
#$ -M hringbauer@ist.ac.at
#$ -N "SecondarySimB"
#$ -m ea
#$ -l mf=16g
#$ -l mem_free=16g
#$ -l h_vmem=16g
#$ -l h_rt=24:00:00
#$ -cwd
#$ -t 1-100:1

export OMP_NUM_THREADS=1
python script_bignick.py $SGE_TASK_ID
