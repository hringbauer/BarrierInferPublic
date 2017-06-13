#!/bin/sh
# Simple Loop that calls all my Bash Scripts.
# Loop 10x10 Times; and for every batch wait.

for i in `seq 3 3`; # from lower to upper: Including end points
do 
   for j in `seq 5 20`;
   do
      echo "run" $i "Sub Run" $j ":" $(($i * 10 + $j)) &
      python serial_script.py $(($i * 10 + $j)) > "output_mg2.txt" 2> "error_mg2.txt" & # Run Process in Background
   done
   wait  # Waits for everything to finish!
   #python serial_script.py $i > "output"$i".txt" 2> "error2"$i".txt" & # Run Process in Background
done
