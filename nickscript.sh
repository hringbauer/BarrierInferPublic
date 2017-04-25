#!/bin/bash
#
#$ -S /bin/bash
#$ -v TST=abc
#$ -M hringbauer@ist.ac.at
#$ -N "BarrierArrayHarald3"
#$ -m ea
#$ -l mf=10G
#$ -l mem_free=16g
#$ -l h_vmem=20g
#$ -l h_rt=12:00:00
#$ -cwd
#$ -t 1-100:1

export OMP_NUM_THREADS=1
python script_bignick.py $SGE_TASK_ID
