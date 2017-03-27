'''
Script for Multi-Runs.
Here you go Mr. AK - make it happen:-)
'''

from multi_run import fac_method

folder = "./nbh_folder/"   # Where the results are saved to.
mp = 1                     # Whether to use MultiProcessing. 0: No 1: Yes

MultiRun = fac_method("multi_nbh", folder, multi_processing = mp)  # Loads the right class.

########### For creating the data sets ###############
'''Mr. Ak: Run the create_data_set thing again for 1 and 7. Probably delete the genotype and position file 
from the folder before so it saves cleanly.'''
#data_set_nr = int(sys.argv[i])  # Which data-set to use
data_set_nr = 30
#MultiRun.create_data_set(data_set_nr)     # Creates data set and saves to Folder.





########### For analyzing the data sets###############
'''Mr. AK: Run this for data-sets 10-12, 40-42, 60-62, and 80-82.
One time for method=1 and one time for method=2'''
i = int(sys.argv[i])           # Which data-set to use
MultiRun.analyze_data_set(data_set_nr, method=1)   # Analyzes the results and pickles them.



















