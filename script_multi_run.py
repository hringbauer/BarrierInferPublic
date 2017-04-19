'''
Script for Multi-Runs.
Here you go Mr. AK - make it happen:-)
'''

from multi_run import fac_method
import cPickle as pickle
import sys


#folder = "./nbh_folder/"  # Where the results are saved to.
folder = "./barrier_folder1/"
mp = 1  # Whether to use MultiProcessing. 0: No 1: Yes

#MultiRun = fac_method("multi_nbh", folder, multi_processing=mp)  # Loads the right class.
''''Mr AK - just use this:'''
MultiRun = fac_method("multi_barrier", folder, multi_processing=1) 


########### For creating the data sets ###############
#data_set_nr = int(sys.argv[1])  # Which data-set to use
#data_set_nr = 1
#MultiRun.create_data_set(data_set_nr)     # Creates data set and saves to Folder.

########### For analyzing the data sets###############
'''Mr. AK: Run this for data-sets 10-12, 40-42, 60-62, and 80-82.
One time for method=1 and one time for method=2'''
#data_set_nr = int(sys.argv[1])  # Which data-set to use
#MultiRun.analyze_data_set(data_set_nr, method=1)
#MultiRun.analyze_data_set(data_set_nr, method=2)  # Analyzes the results and pickles them.



###########Methods for creating Barrier Likelihood Profiles:
def load_pickle_data(i):
    '''Function To load pickled Data.
    Also visualizes it.'''
    data_folder = folder
    # path = data_folder + "result" + str(i).zfill(2) + ".p"  # Path to Alex Estimates
    
    # subfolder_meth = "estimate" + str(2) + "/"  # Path to binned Estimates
    # path = self.data_folder + subfolder_meth + "result" + str(i).zfill(2) + ".p"
    
    # Coordinates for more :
    subfolder_meth = "method2" + "/"  # Sets subfolder to which Method to use.
    path = data_folder + subfolder_meth + "result" + str(i).zfill(2) + ".p"
    
    res = pickle.load(open(path, "rb"))  # Loads the Data
    return res

MultiRun.barrier_ll(data_set_nr, nbh=nbh, L=l, t0=0.5, random_ind_nr=1000, position_barrier=500.5, barrier_strengths=21)

















