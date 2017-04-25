'''
Script for Multi-Runs.
Here you go Mr. AK - make it happen:-)
'''

from multi_run import fac_method
import pickle as pickle
import numpy as np
import sys

mp = 0  # Whether to use MultiProcessing. 0: No 1: Yes

        
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


#MultiRun.barrier_ll(data_set_nr, nbh=5*4*3.141, L=0.006, t0=0.5, random_ind_nr=1000, position_barrier=500.5, barrier_strengths=21)
#MultiRun.barrier_ll(data_set_nr, nbh=nbh, L=l, t0=0.5, random_ind_nr=1000, position_barrier=500.5, barrier_strengths=21)
def analyze_barrier_strengths_ll():
    '''Method to analyze Barrier strengths ll using fits from other methods.'''
    for data_set_nr in xrange(100):
        nbh, l, _, _ = load_pickle_data(data_set_nr)[0]
        #MultiRun.barrier_ll(data_set_nr, nbh=5*4*3.141, L=0.006, t0=0.5, random_ind_nr=1000, position_barrier=500.5, barrier_strengths=21)
        MultiRun.barrier_ll(data_set_nr, nbh=nbh, L=l, t0=0.5, random_ind_nr=1000, position_barrier=500.5, barrier_strengths=21)


#if __name__ == "__main__":
    #analyze_barrier_strengths_ll()
    #analyze_nbh_data_sets_model(data_set_nrs=[2,]) # Analyze 100 Neighborhood Samples
    


data_set_nr = int(sys.argv[1])  # Which data-set to use
#data_set_nr = 29
print("Starting Dataset Nr %i:" % data_set_nr)
data_set_nr = data_set_nr - 1

folder = "./nbh_folder_gauss/"  # Where the results are saved to.
    
#MultiRun = fac_method("multi_nbh", folder, multi_processing=mp)  # Loads the right class.
MultiRun = fac_method("multi_nbh_gaussian", folder, multi_processing=mp) 

########### For creating the data sets ###############

#MultiRun.create_data_set(data_set_nr)     # Creates data set and saves to Folder.
#MultiRun.analyze_data_set(data_set_nr, method=2)
MultiRun.analyze_data_set(data_set_nr, method=0)  # Analyzes the results and pickles them.
#MultiRun.analyze_data_set(data_set_nr, method=1)

print("Run %i completed. Good job!" % data_set_nr)









