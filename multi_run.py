'''
Created on March 5th, 2017:
@Harald: Contains methods for producing as well as
analyzing multiple data-sets. There is one Master Class, and all other classes inherit from it.
Every sub class sets the properties of the analysis and estimation object. 
Plotting is usually done from the inherited method though.
'''

from statsmodels.base.model import GenericLikelihoodModel
from kernels import fac_kernel
from time import time
from grid import Grid
import matplotlib.pyplot as plt
import numpy as np
import os
    

class MultiRun(object):
    '''
    Class that can produce as well as analyze multiple barrier datasets.
    '''
    data_folder = ""  # Folder
    file_name = ""  # Which name to use for the files. 
    data_set_number = 0  # Number of the datasets
    pickle_path = ""  # File Name where the Array of Results and Uncertainties are pickled

    
    def __init__(self, data_folder="./"):
        '''
        Constructor. Sets the Grid class to produce and the MLE class to analyze the Data
        '''
        self.data_folder = data_folder
        
    def set_grid_params(self, grid, run_nr):
        '''Sets the Parameters for the grid depending on the run-nr'''
        raise NotImplementedError("Implement this!")
        
    def set_mle_object(self):
        '''Analyzes Data Set'''
        raise NotImplementedError("Implement this!")
    
    def create_data_set(self):
        '''Method to create the data set'''
        raise NotImplementedError("Implement this!")
    
    def analyze_data_sets(self, coords, genotype_mat):
        '''Method that analyzes a single data set with MLE.
        Return Parameter estimates and 95% Uncertainties'''
        raise NotImplementedError("Implement this!")
    
    def plot_estimates(self):
        '''Method to plot estimates and uncertainties'''
        raise NotImplementedError("Implement this!")
    
    def save_data_set(self, coords, genotype_matrix, data_set_nr):
        '''Method to save a Data set'''
        data_set_name = self.data_folder + self.name + str(data_set_nr).zfill(2) + ".csv"
        
        # Check whether Directory exists and creates it if necessary
        directory = os.path.dirname(data_set_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        np.savetxt(data_set_name, coords, delimiter="$")  # Save the coordinates
        np.savetxt(data_set_name, genotype_matrix, delimiter="$")  # Save the data 
        
    def load_data_set(self, data_set_nr):
        '''Loads the data set'''
        data_set_name = self.data_folder + self.name + str(data_set_nr).zfill(2) + ".csv"
        position_list = np.loadtxt(data_set_name, delimiter='$').astype('float64')
        genotype_matrix = np.loadtxt(data_set_name, delimiter='$').astype('float64')
        return position_list, genotype_matrix
    
    def pickle_parameters(self, parameters_names, parameter_values, other_info):
        '''Pickles all parameters that have been used into the folder.
        Should be called in create_data_sets
        For later information'''
        path = self.data_folder + "parameters.p"
        pickle.dump((parameters_names, parameter_values, other_infos), open(path, "wb"))  # Pickle the Info
    
class TestRun(MultiRun):
    '''First simple class to test whether everything works'''
    def __init__(self, folder, **kwds):
        super(TestRun, self).__init__(**kwds)  # Run initializer of full MLE object.
        self.name = "test_file"
        self.data_folder = folder
        
        
    def create_data_set(self):
        '''Create a Data_Set'''
        # First set all the Parameter Values:
        position_list = np.array([(500 + i, 500 + j) for i in range(-19, 20, 2) for j in range(-25, 25, 2)])
        nr_loci = 2
        t = 10000
        gridsize_x, gridsize_y = 1000, 1000
        sigma = 1.98
        ips = 10  # Number of haploid Individuals per Node (For D_e divide by 2)
        mu=0.001  # Mutation Rate
        ss = 0.1
        p_delta = np.random.normal(scale=ss, size=nr_loci)  # Draw some random Delta p from a normal distribution
        p_mean = np.ones(nr_loci) * 0.5  # Sets the mean allele Frequency
        p_mean = p_mean + p_delta
        
        # print("Observed Standard Deviation: %.4f" % np.std(p_delta))
        # print("Observed Sqrt of Squared Deviation: %f" % np.sqrt(np.mean(p_delta ** 2)))
        
        for j in xrange(10):  # Create 10 Data-Sets
            genotype_matrix = np.zeros((len(position_list), nr_loci))  # Set Genotype Matrix to 0
            
            for i in range(nr_loci):
                grid = Grid()  # Creates new Grid. Maybe later on use factory Method
                grid.set_parameters(gridsize_x, gridsize_y, sigma, ips, mu)
                print("Doing run: %i, Simulation: %i " % (j, i))
                grid.set_samples(position_list)
                grid.update_grid_t(t, p=p_mean[i])  # Uses p_mean[i] as mean allele Frequency.
                genotype_matrix[:, i] = grid.genotypes
            self.save_data_set(position_list, genotype_matrix, j)
            
        # Now Pickle Some addtional Information
        p_names = ["Position List", "Nr Loci", "t", "p_mean", "sigma", "mu", "ips", "ss"]
        ps = [position_list, nr_loci, t, p_mean, sigma, mu, ips , ss]
        additional_info = ("10 Test Runs for Grid object with 10 loci")
        self.pickle_parameters(p_names, ps, additional_info)
            
    def analyze_data_sets(self):
        '''Analyzes a single Data-Set'''
        print("To Implement")
        

def fac_method(method, folder):
    '''Factory Method to give the right Class which creates and analyzes the data-set'''
    if method == "testrun":
        return TestRun(folder)
        
#######################
# Some methods to test & and run this class:
######################### Some lines to test the code and make plots
if __name__ == "__main__":
    Test = fac_method("testrun", "./testfolder/")
    Test.create_data_set()


