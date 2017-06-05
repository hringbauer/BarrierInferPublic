#!/bin/sh
# Simple Loop that calls all my Bash Scripts.
for i in `seq 0 99`; # from lower to upper: Including end points
do 
   python serial_script.py $i > "output"$i".txt" 2> "error2"$i"."txt" & # Run Process in Background
done
