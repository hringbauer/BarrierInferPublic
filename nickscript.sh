#!/bin/bash
#
#$ -S /bin/bash
#$ -v TST=abc
#$ -M hringbauer@ist.ac.at
#$ -N "SecondarySimB"
#$ -m ea
#$ -l mf=8g
#$ -l mem_free=8g
#$ -l h_vmem=8g
#$ -l h_rt=24:00:00
#$ -cwd
#$ -t 1-460:1

export OMP_NUM_THREADS=1
module load tensorflow/python-2.7/1.1.0
python script_bignick.py $SGE_TASK_ID
