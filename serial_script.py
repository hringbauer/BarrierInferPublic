'''
Script to do serial Runs on the cluster.
Loads everything and then does one run after another.
'''

from multi_run import fac_method
import pickle as pickle
import numpy as np
import sys

mp = 0  # Whether to use MultiProcessing. 0: No 1: Yes

# data_set_nr = int(sys.argv[1])  # Which data-set to use

# folder = "./nbh_folder/"
# folder = "./nbh_folder_gauss/"  # Where the results are saved to.
# folder = "./cluster_folder/"
# folder = "./bts_folder_test/"
# folder = "./hz_folder/"
# folder = "./multi_ind_nr/"
folder = "./barrier_folder2/"
# folder ="./multi_loci_nr/"
# folder = "./multi_2nd/"
# folder = "./multi_2nd_b/"
# folder = "./multi_barrier_synth/"

# MultiRun = fac_method("multi_nbh", folder, multi_processing=mp)  # Loads the right class.
# MultiRun = fac_method("multi_nbh_gaussian", folder, multi_processing=mp) 
# MultiRun = fac_method("multi_cluster", folder, multi_processing=1)   
# MultiRun = fac_method("multi_bts", folder, multi_processing=1)
# MultiRun = fac_method("multi_HZ", folder, multi_processing=mp)
#MultiRun = fac_method("multi_inds", folder, multi_processing=mp)
MultiRun = fac_method("multi_barrier", folder, multi_processing=mp)
# MultiRun = fac_method("multi_2nd_cont", folder, multi_processing=mp)
# MultiRun = fac_method("multi_barrier_pos", folder, multi_processing=mp)

# MultiRun = fac_method("multi_loci", folder, multi_processing=mp)



########### For creating and analyzing the data sets ###############
# MultiRun.create_data_set(data_set_nr)     # Creates data set and saves to Folder.
# MultiRun.create_data_set(data_set_nr, barrier_strength=0.05)


########### For analyzing the data set ##############
# MultiRun.analyze_data_set(data_set_nr, position_barrier=2.0, res_folder="barrier3/" ,method=2)  # Position Barrier is there for the HZ Data.
# MultiRun.analyze_data_set(data_set_nr, position_barrier=-20.0, res_folder="barrier20m/" ,method=2)
# MultiRun.analyze_data_set(data_set_nr, position_barrier=18.0, res_folder="barrier18m/" ,method=2)
# MultiRun.analyze_data_set(data_set_nr, method=0)  # Analyzes the results and pickles them.
# MultiRun.analyze_data_set(data_set_nr, method=1)

#for i in xrange(1, 99):
data_set_nr = int(sys.argv[1])  # Which data-set to use
# data_set_nr = 50
print("Do Run Nr. %i" % data_set_nr)
MultiRun.analyze_data_set(data_set_nr, method=0)
# MultiRun.analyze_data_set_cleaning(data_set_nr, method=2)

print("Run %i completed. Good job!" % data_set_nr)



