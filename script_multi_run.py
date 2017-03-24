'''
Script for Multi-Runs.
Here you go Mr. AK - make it happen:-)
'''

from multi_run import fac_method

folder = "./nbh_folder/"   # Where the results are saved to.
mp = 1 # Whether to use MultiProcessing. 0: No 1: Yes

MultiRun = fac_method("multi_nbh", folder, multi_processing = mp)  # Loads the right class.

########### For creating the data sets ###############
#data_set_nr = int(sys.argv[i])  # Which data-set to use
data_set_nr = 30
#MultiRun.create_data_set(data_set_nr)     # Creates data set and saves to Folder folder





########### For analyzing the data sets###############
#i = int(sys.argv[i])           # Which data-set to use
#MultiRun.analyze_data_set(data_set_nr)   # Analyses the results and pickles them.

for i in xrange(100):
    MultiRun.analyze_data_set(i, mle_pw=2)   # Analyses the results and pickles them.

















