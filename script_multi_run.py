'''
Script for Multi-Runs.
Here you go Mr. AK - make it happen:-)
'''

from multi_run import fac_method

folder = "./nbh_folder/"   # Where the results are saved to.

MultiRun = fac_method("multi_nbh", folder)  # Loads the right class.

########### For creating the data sets ###############
data_set_nr = int(sys.argv[i])  # Which data-set to use
MultiRun.create_data_set(i)     # Creates data set and saves to Folder folder





########### For analyzing the data sets###############
#i = int(sys.argv[i])           # Which data-set to use
#MultiRun.analyze_data_set(i)   # Analysises the results and pickles them.

















