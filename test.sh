#!/bin/sh
# Simple Loop that calls all my Bash Scripts.
for i in `seq 0 50`; # from lower to upper: Including end points
do 
   python serial_script.py $i & # Run Process in Background
done
