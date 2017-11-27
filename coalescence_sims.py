###
# Code to run Coalescence Simulations on the Cluster
###
# Copy/Paste from Jupyter Here!

from grid import Coalescence_Grid  # Import the important Grid Class
from grid import Laplace_Offsets
import numpy as np 
import matplotlib.pyplot as plt # For plotting stuff
import cPickle as pickle  # Used for pickling the intermediate Results
from scipy.special import erfc
from time import time
import sys  # Import for Variables

data_folder="coal_times/"  # Where to save the Results to

# Set up the Grid
# and Define the Grid Parameters:
drawer = Laplace_Offsets(10000, sigma=0.965)
grid = Coalescence_Grid(drawer = drawer)
grid.gridsize_x = 40
grid.gridsize_y = 40
grid.start_pos1 = [8, 5]  # Position of first Individual
grid.start_pos2 = [12, 5]  # Position of second Individual
grid.ips=10
grid.barrier=19.5
grid.barrier_strength=1.0


def do_panel_simulations(pos1, pos2, save_path, short=False, replicates=100, report_int=1,
                        gamma1=1.0, gamma2=0.01):
    '''Do all the simualtions for a panel.
    report_int: When to report.
    pos1, pos2: Postions of the Individuals ([x,y])
    gamma: The barrier strengths'''
    
    # Do the simulations
    if short == True:
        grid.t_max = 1000
        pre_fix = "short_"
    elif short == False:
        grid.t_max = 200000
        pre_fix = "long_"
        
    grid.start_pos1 = pos1  # Position of first Individual
    grid.start_pos2 = pos2  # Position of second Individual
    
    print("Doing Simulations for Inds at:")
    print(pos1)
    print(pos2)

    # Do the Grid without a Barrier:
    grid.barrier_strength=gamma1
    coal_times = grid.return_coalescence_times(n=replicates, report_int=report_int) # Do n runs

    # Set Grid Parameters to Barrier:
    grid.barrier_strength=gamma2
    coal_times_bar = grid.return_coalescence_times(n=replicates, report_int=report_int)
    print("Simulations complete!")
    
    path=data_folder + pre_fix + save_path   #"coal89.csv"
    path_bar=data_folder + pre_fix + "barrier_" + save_path
    
    # Remove the NANs:
    coal_times = coal_times[~np.isnan(coal_times)]
    coal_times_bar = coal_times_bar[~np.isnan(coal_times_bar)]
    
    
    # Save the simulations
    np.savetxt(path, coal_times, fmt='%i')  # Save the additional Info
    np.savetxt(path_bar, coal_times_bar, fmt='%i') # Save the Coalescence with Barrier
    print("Successfully Saved.")
    
    return coal_times, coal_times_bar # Return the Coalescence Time Vectors



def do_panel_row_simulations(row=0, report_int=50, reps_short=100, reps_long=1000):
    '''Does the simulations for'''
    positions1 = [[18,20], [12,20], [18,20]]
    positions2 = [[18,20], [12,20], [21,20]]
    save_names = ["1818.csv","1212.csv","1821.csv"]
    replicates_short = 1000  # Do replicates for long.
    replicates_long = 100  # Do replicates for short.
    
    pos1 = positions1[row]
    pos2 = positions2[row]
    save_name = save_names[row]   
    
    start = time()
    do_panel_simulations(pos1=pos1, pos2=pos2, save_path=save_name,
                         replicates=replicates_short, report_int=report_int, short=True)
    end = time()
    print("Time Run Short: %.4f" % (end-start))
    
    start = time()
    do_panel_simulations(pos1=pos1, pos2=pos2, save_path=save_name,
                     replicates=replicates_long, report_int=report_int, short=False)
    end = time()
    print("Time Run Long: %.4f" % (end-start))
    
# Which Row to run
#row=0
row = int(sys.argv[1])  # Read the input
do_panel_row_simulations(row=row, reps_short=1000, reps_long=100, report_int=100)
print("Run %i completed. Good job!" % row)
