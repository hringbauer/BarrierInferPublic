#!/bin/sh
# Simple Loop that calls all my Bash Scripts.
# Loop 10x10 Times; and for every batch wait.
# Mainly used to call TensorFlow within tmux

for i in `seq 3 3`; # from lower to upper: Including end points
do 
for j in `seq 10 20`;
   do
    echo "run" $i "Sub Run" $j ":" $(($i * 20 + $j)) &
    python serial_script.py $(($i * 20 + $j)) > "output_mi70.txt" 2> "error_mi70.txt" & # Run Process in Background
    echo "run" $j "Done!" &
   done
   wait  # Waits for everything to finish!
   #python serial_script.py $i > "output"$i".txt" 2> "error2"$i".txt" & # Run Process in Background
done
