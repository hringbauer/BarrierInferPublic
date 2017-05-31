#!/bin/bash
#
#$ -S /bin/bash
#$ -v TST=abc
#$ -M hringbauer@ist.ac.at
#$ -N "TestSerial"
#$ -m ea
#$ -l mf=48g
#$ -l mem_free=48g
#$ -l h_vmem=48g
#$ -l h_rt=48:00:00
#$ -cwd

export OMP_NUM_THREADS=1
module load tensorflow/python-2.7/1.1.0
python serial_script.py
