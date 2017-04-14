#!/bin/bash
#
#$ -S /bin/bash
#$ -v TST=abc
#$ -M hringbauer@ist.ac.at
#$ -N "BarrierSimulationsHarald"
#$ -m ea
#$ -l mf=50G
#$ -l h_vmem=500G
#$ -l h_rt=20:00:00
#$ -cwd

export OMP_NUM_THREADS=1
python script_bignick.py
