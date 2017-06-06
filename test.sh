#!/bin/sh
# Simple Loop that calls all my Bash Scripts.


for i in `seq 0 10`; # from lower to upper: Including end points
do 
   for j in `seq 0 10`;
   do
      #echo "run" $i "Sub Run" $j ":" $((($i - 1) * 10 + ($j - 1))) &
      python serial_script.py $((($i - 1) * 10 + ($j - 1))) > "output.txt" 2> "error.txt" & # Run Process in Background
   done
   wait  # Waits for everything to finish!
   #python serial_script.py $i > "output"$i".txt" 2> "error2"$i".txt" & # Run Process in Background
done
