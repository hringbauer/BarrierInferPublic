#!/bin/bash
#
#$ -S /bin/bash
#$ -v TST=abc
#$ -M hringbauer@ist.ac.at
#$ -N "BarrierArrayHarald2"
#$ -m ea
#$ -l mf=10G
#$ -l h_vmem=10G
#$ -l h_rt=8:00:00
#$ -cwd
#$ -t 1-100:1

export OMP_NUM_THREADS=1
python script_bignick.py $SGE_TASK_ID
