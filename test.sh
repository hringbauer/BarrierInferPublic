#!/bin/sh
# Simple Loop that calls all my Bash Scripts.
for i in `seq 1 10`;
do 
   python your_script.py $i
done
