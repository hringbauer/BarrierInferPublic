'''
Script for Multi-Runs.
Here you go Mr. AK - make it happen:-)
'''

from multi_run import fac_method

import sys


#folder = "./nbh_folder/"  # Where the results are saved to.
folder = "./barrier_folder1/"
mp = 1  # Whether to use MultiProcessing. 0: No 1: Yes

#MultiRun = fac_method("multi_nbh", folder, multi_processing=mp)  # Loads the right class.
''''Mr AK - just use this:'''
MultiRun = fac_method("multi_barrier", folder, multi_processing=1) 

########### For creating the data sets ###############
data_set_nr = int(sys.argv[1])  # Which data-set to use
#data_set_nr = 1
MultiRun.create_data_set(data_set_nr)     # Creates data set and saves to Folder.

########### For analyzing the data sets###############
'''Mr. AK: Run this for data-sets 10-12, 40-42, 60-62, and 80-82.
One time for method=1 and one time for method=2'''
#data_set_nr = int(sys.argv[1])  # Which data-set to use
#MultiRun.analyze_data_set(data_set_nr, method=1)
MultiRun.analyze_data_set(data_set_nr, method=2)  # Analyzes the results and pickles them.
    



















