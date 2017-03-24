'''
Created on March 5th, 2017:
@Harald: Contains methods for producing as well as
analyzing multiple data-sets. There is one Master Class, and all other classes inherit from it:
Every MultiRun subclass will require implementation of all methods required for analysis.
Every class saves single result runs to numbered files in a folder - and loads them from there for analysis.
Analysis is then conducted and results/uncertainty estimates (confidence intervalls) are pickled.
For visualization the pickled file is loaded then.
'''

from statsmodels.base.model import GenericLikelihoodModel
from kernels import fac_kernel
from time import time
from grid import Grid
from mle_class import MLE_estimator
from mle_pairwise import MLE_pairwise
from random import shuffle 
from analysis import Analysis
import matplotlib.pyplot as plt
import numpy as np
import os
import cPickle as pickle
    

class MultiRun(object):
    '''
    Class that can produce as well as analyze multiple barrier datasets.
    '''
    data_folder = ""  # Folder
    file_name = "run"  # Which name to use for the files. 
    param_estimates = 0  # Will be matrix for parameters
    uncert_estimates = 0  # Will be matrix for parameter uncertainties
    nr_data_sets = 0  # Number of the datasets
    multi_processing = 0  # Whether to actually use multi-processing

    
    def __init__(self, data_folder, nr_data_sets, nr_params, multi_processing=0):
        '''
        Constructor. Sets the Grid class to produce and the MLE class to analyze the Data
        '''
        self.data_folder = data_folder
        self.nr_data_sets = nr_data_sets
        self.param_estimates = np.zeros((nr_data_sets, nr_params))  # Creates array for Parameter Estimates
        self.uncert_estimates = np.zeros((nr_data_sets, nr_params * 2))  # Creates array for Uncertainty Estimates
        self.multi_processing = multi_processing 
        
    def set_grid_params(self, grid, run_nr):
        '''Sets the Parameters for the grid depending on the run-nr'''
        raise NotImplementedError("Implement this!")
        
    def set_mle_object(self):
        '''Analyzes Data Set'''
        raise NotImplementedError("Implement this!")
    
    def create_data_set(self, data_set_nr):
        '''Method to create data_set nr data_set_nr.'''
        raise NotImplementedError("Implement this!")
    
    def analyze_data_set(self, data_set_nr):
        '''Method that analyze data-set i.
        Return Parameter estimates and 95% Uncertainties'''
        raise NotImplementedError("Implement this!")
    
    def visualize_results(self):
        '''Method to visualize the results and uncertainty estimates'''
        raise NotImplementedError("Implement this!")
    
    def create_all_data_sets(self):
        '''Method that creates all data sets.
        Could be parallelized'''
        for i in xrange(self.nr_data_sets):
            self.create_data_set(i)
        
    def analyze_all_data_sets(self):
        '''Method that analyzes all data sets.
        Could be parallelized'''
        for i in xrange(self.nr_data_sets):
            self.analyze_data_set(i)
    
    def plot_estimates(self):
        '''Method to plot estimates and uncertainties'''
        raise NotImplementedError("Implement this!")
    
    def save_data_set(self, coords, genotype_matrix, data_set_nr):
        '''Method to save a Data set'''
        data_set_name_coords = self.data_folder + self.name + "_coords" + str(data_set_nr).zfill(2) + ".csv"
        data_set_name_genotypes = self.data_folder + self.name + "_genotypes" + str(data_set_nr).zfill(2) + ".csv"
        # Check whether Directory exists and creates it if necessary
        directory = os.path.dirname(data_set_name_coords)  # Extract Directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        np.savetxt(data_set_name_coords, coords, delimiter="$")  # Save the coordinates
        np.savetxt(data_set_name_genotypes, genotype_matrix, delimiter="$")  # Save the data 
        
    def load_data_set(self, data_set_nr):
        '''Loads the data set'''
        data_set_name_coords = self.data_folder + self.name + "_coords" + str(data_set_nr).zfill(2) + ".csv"
        data_set_name_genotypes = self.data_folder + self.name + "_genotypes" + str(data_set_nr).zfill(2) + ".csv"
        

        position_list = np.loadtxt(data_set_name_coords, delimiter='$').astype('float64')
        genotype_matrix = np.loadtxt(data_set_name_genotypes, delimiter='$').astype('float64')
        return position_list, genotype_matrix
    
    def pickle_parameters(self, parameters_names, parameter_values, other_info):
        '''Pickles all parameters that have been used into the folder.
        Should be called in create_data_sets
        For later information'''
        path = self.data_folder + "parameters.p"
        pickle.dump((parameters_names, parameter_values, other_info), open(path, "wb"))  # Pickle the Info
        
    def save_analysis(self):
        '''Pickles the Outcome of the Analysis. For later information. '''
        path = self.data_folder + "analysis.p"
        pickle.dump((self.param_estimates, self.uncert_estimates), open(path, "wb"))
        
    def load_analysis(self):
        '''Loads Results.'''
        path = self.data_folder + "analysis.p"
        (self.param_estimates, self.uncert_estimates) = pickle.load(open(path, "rb"))
        return self.param_estimates, self.uncert_estimates

###############################################################################################################################

class MultiNbh(MultiRun):
    '''First simple class to test whether everything works.
    The Full Goal is to find out at which Neighborhood Size the method to estimate IBD works best.
    Everything set so that 100 Data-Sets are run.'''
    def __init__(self, folder, nr_data_sets=100, nr_params=4, **kwds):
        super(MultiNbh, self).__init__(folder, nr_data_sets, nr_params, **kwds)  # Run initializer of full MLE object.
        self.name = "nbh_file"
        # self.data_folder = folder
        
    def create_data_set(self, data_set_nr):
        '''Create a Data_Set. Override method.'''
        print("Craeting Dataset: %i" % data_set_nr)
        # First set all the Parameter Values:
        ips_list = 25 * [2.0] + 25 * [10.0] + 25 * [18.0] + 25 * [26.0]
        ips = ips_list[data_set_nr]  # Number of haploid Individuals per Node (For D_e divide by 2)  Loads the right Neighborhood Size
        
        
        position_list = np.array([(500 + i, 500 + j) for i in range(-19, 21, 2) for j in range(-49, 51, 2)])  # 1000 Individuals; spaced 2 sigma apart.
        nr_loci = 200
        t = 5000
        gridsize_x, gridsize_y = 1000, 1000
        sigma = 0.965  # 0.965 # 1.98
        mu = 0.003  # Mutation/Long Distance Migration Rate # Idea is that at mu=0.01 there is quick decay which stabilizes at around sd_p
        sd_p = 0.1
        p_delta = np.random.normal(scale=sd_p, size=nr_loci)  # Draw some random Delta p from a normal distribution
        p_mean = np.ones(nr_loci) * 0.5  # Sets the mean allele Frequency
        p_mean = p_mean + p_delta
        
        # print("Observed Standard Deviation: %.4f" % np.std(p_delta))
        # print("Observed Sqrt of Squared Deviation: %f" % np.sqrt(np.mean(p_delta ** 2)))
        
        genotype_matrix = np.zeros((len(position_list), nr_loci))  # Set Genotype Matrix to 0
        
        for i in range(nr_loci):
            grid = Grid()  # Creates new Grid. Maybe later on use factory Method
            grid.set_parameters(gridsize_x, gridsize_y, sigma, ips, mu)
            print("Doing data set: %i, Simulation: %i " % (data_set_nr, i))
            grid.set_samples(position_list)
            grid.update_grid_t(t, p=p_mean[i])  # Uses p_mean[i] as mean allele Frequency.
            genotype_matrix[:, i] = grid.genotypes
        self.save_data_set(position_list, genotype_matrix, data_set_nr)
        
            
        # Now Pickle Some additional Information:
        p_names = ["Nr Loci", "t", "p_mean", "sigma", "mu", "ips", "sd_p", "Position List"]
        ps = [nr_loci, t, p_mean, sigma, mu, ips, sd_p, position_list]
        additional_info = ("1 Test Run for Grid object with high neighborhood size")
        self.pickle_parameters(p_names, ps, additional_info)
            
    def analyze_data_set(self, data_set_nr, random_ind_nr=1000, mle_pw=0):
        '''Create Data Set. Override Method. mle_pw: Whether to use Pairwise Liklihood'''
        position_list, genotype_mat = self.load_data_set(data_set_nr)  # Loads the Data 
        
        # Creates the "right" starting parameters:
        ips_list = 25 * [2.0] + 25 * [10.0] + 25 * [18.0] + 25 * [26.0]
        ips_list = np.array(ips_list)
        nbh_sizes = ips_list / 2.0 * 4 * np.pi  # 4 pi sigma**2 D = 4 * pi * 1 * ips/2.0
        start_list = [[nbh_sizes, 0.005, 0.04] for nbh_sizes in nbh_sizes]  # General Vector for Start-Lists
        
        # Pick Random_ind_nr many Individuals:
        inds = range(len(position_list))
        shuffle(inds)  # Random permutation of the indices. If not random draw - comment out
        inds = inds[:random_ind_nr]  # Only load first nr_inds

        position_list = position_list[inds, :]
        genotype_mat = genotype_mat[inds, :]
        
        if mle_pw==0:
            MLE_obj = MLE_estimator("DiffusionK0", position_list, genotype_mat, multi_processing=self.multi_processing) 
        elif mle_pw==1:
            MLE_obj = MLE_pairwise("DiffusionK0", position_list, genotype_mat, multi_processing=self.multi_processing)
            start_list = [[nbh_sizes, 0.006, 0.01] for nbh_sizes in nbh_sizes]  # Update Vector of Start Lists
        elif mle_pw==2: # Do the fitting based on binned data
            MLE_obj = Analysis(position_list, genotype_mat) 
            
            
        else: raise ValueError("Wrong Input for mle_pw")
        
        fit = MLE_obj.fit(start_params=start_list[data_set_nr])

        params = fit.params
        conf_ind = fit.conf_int()
        
        # Pickle Parameter Estimates:
        subfolder_meth="estimate" + str(mle_pw) + "/"
        path=self.data_folder + subfolder_meth + "result" + str(data_set_nr).zfill(2) + ".p"
        
        directory = os.path.dirname(path)  # Extract Directory
        if not os.path.exists(directory):  # Creates Folder if not already existing
            os.makedirs(directory)
            
        pickle.dump((params, conf_ind), open(path, "wb"))  # Pickle the Info

        
        
        
        
    def visualize_results(self):
        '''Load and visualize the Results'''
        param_estimates, uncert_estimates = self.load_analysis()  # Loads and saves Parameter Estimates and Uncertainty estimates
        
        plt.figure()
        plt.errorbar()  # Fully Implement this plotting.
        plt.show()
        
    def temp_visualize(self):
        '''Temporary Function to plot the Estimates
        that were run on cluster.'''
        
        
        # First quick function to unpickle the data:
        def load_pickle_data(i, arg_nr):
            '''Function To load pickled Data.
            Also visualizes it.'''
            data_folder = self.data_folder
            #path = data_folder + "result" + str(i).zfill(2) + ".p"  # Path to Alex Estimates
            
            subfolder_meth="estimate" + str(2) + "/"  # Path to binned Estimates
            path=self.data_folder + subfolder_meth + "result" + str(i).zfill(2) + ".p"
            
            res = pickle.load(open(path, "rb"))  # Loads the Data
            return res[arg_nr]
        
        res_numbers = range(0, 7) + range(8, 67)
        res_vec = np.array([load_pickle_data(i, 0) for i in res_numbers])
        unc_vec = np.array([load_pickle_data(i, 1) for i in res_numbers])
        
        for i in res_numbers[:-1]:
            print("\nRun: %i" % i)
            for j in range(3):
                print("Parameter: %i" % j)
                print("Value: %f (%f,%f)" % (res_vec[i, j], unc_vec[i, j, 0], unc_vec[i, j, 1]))
                
        
        
        # plt.figure()
        f, ((ax1, ax2, ax3)) = plt.subplots(3, 1, sharex=True)
        
        ax1.hlines(4 * np.pi, 0, 25, linewidth=2, color="r")
        ax1.hlines(4*np.pi * 5, 25, 50, linewidth=2, color="r")
        ax1.hlines(4*np.pi * 9, 50, 66, color="r")
        ax1.errorbar(res_numbers, res_vec[:, 0], yerr=res_vec[:, 0] - unc_vec[:, 0, 0], fmt="bo", label="Nbh")
        ax1.set_ylabel("Nbh", fontsize=18)
        #ax1.legend()
        
        ax2.errorbar(res_numbers, res_vec[:, 1], yerr=res_vec[:, 1] - unc_vec[:, 1, 0], fmt="go", label="L")
        ax2.hlines(0.006, 0, 66, linewidth=2)
        ax2.set_ylabel("L", fontsize=18)
        #ax2.legend()
        
        ax3.errorbar(res_numbers, res_vec[:, 2], yerr=res_vec[:, 2] - unc_vec[:, 2, 0], fmt="ko", label="ss")
        ax3.hlines(0.04, 0, 66, linewidth=2)
        ax3.set_ylabel("SS", fontsize=18)
        #ax3.legend()
        plt.xlabel("Dataset")
        plt.show()
        
    
        
###############################################################################################################################
        
class TestRun(MultiRun):
    '''First simple class to test whether everything works'''
    def __init__(self, folder, nr_data_sets=1, nr_params=4, **kwds):
        super(TestRun, self).__init__(folder, nr_data_sets, nr_params, **kwds)  # Run initializer of full MLE object.
        self.name = "test_file"
        # self.data_folder = folder
        
    def create_data_set(self, data_set_nr):
        '''Create a Data_Set. Override method.'''
        print("Craeting Dataset: %i" % data_set_nr)
        # First set all the Parameter Values:
        position_list = np.array([(500 + i, 500 + j) for i in range(-19, 20, 2) for j in range(-25, 25, 2)])
        nr_loci = 100
        t = 5000
        gridsize_x, gridsize_y = 1000, 1000
        sigma = 0.965  # 0.965 # 1.98
        ips = 12  # Number of haploid Individuals per Node (For D_e divide by 2)
        mu = 0.005  # Mutation Rate
        ss = 0.1
        p_delta = np.random.normal(scale=ss, size=nr_loci)  # Draw some random Delta p from a normal distribution
        p_mean = np.ones(nr_loci) * 0.5  # Sets the mean allele Frequency
        p_mean = p_mean + p_delta
        
        # print("Observed Standard Deviation: %.4f" % np.std(p_delta))
        # print("Observed Sqrt of Squared Deviation: %f" % np.sqrt(np.mean(p_delta ** 2)))
        
        genotype_matrix = np.zeros((len(position_list), nr_loci))  # Set Genotype Matrix to 0
        
        for i in range(nr_loci):
            grid = Grid()  # Creates new Grid. Maybe later on use factory Method
            grid.set_parameters(gridsize_x, gridsize_y, sigma, ips, mu)
            print("Doing data set: %i, Simulation: %i " % (data_set_nr, i))
            grid.set_samples(position_list)
            grid.update_grid_t(t, p=p_mean[i])  # Uses p_mean[i] as mean allele Frequency.
            genotype_matrix[:, i] = grid.genotypes
        self.save_data_set(position_list, genotype_matrix, data_set_nr)
        
            
        # Now Pickle Some additional Information:
        p_names = ["Nr Loci", "t", "p_mean", "sigma", "mu", "ips", "ss", "Position List"]
        ps = [nr_loci, t, p_mean, sigma, mu, ips, ss, position_list]
        additional_info = ("1 Test Run for Grid object with high neighborhood size")
        self.pickle_parameters(p_names, ps, additional_info)
            
    def analyze_data_set(self, data_set_nr, random_ind_nr=500):
        '''Create Data Set. Override Method.'''
        position_list, genotype_mat = self.load_data_set(data_set_nr)  # Loads the Data 
        
        # Pick Random_ind_nr many Individuals:
        inds = range(len(position_list))
        shuffle(inds)  # Random permutation of the indices. If not random draw - comment out
        inds = inds[:random_ind_nr]  # Only load first nr_inds

        position_list = position_list[inds, :]
        genotype_mat = genotype_mat[inds, :]
        
        start_params = [100.0, 0.005, 0.038]  # Sets the Start Parameters
        MLE_obj = MLE_estimator("DiffusionK0", position_list, genotype_mat) 
        fit = MLE_obj.fit(start_params=[100.0, 0.01, 0.04])
        
        
        print(fit.summary())
    
        # results0 = ml_estimator.fit(method="BFGS")  # Do the actual fit. method="BFGS" possible
        params = fit.params
        conf_ind = fit.conf_int()
        
        print(np.shape(conf_int))
        print(params)  # Print Parameter Estimates
        print(conf_ind)  # Print Confidence Intervals
        
        
        print("Trying to save. uhuhuh.")
        # Saves the results of Analysis to local object
        self.param_estimates[data_set_nr, :] = fit.params
        self.uncertain_estimates[data_set_nr, :] = fit.conf_int
        
        # Pickles the parameters and confidence intervalls
        path = self.data_folder + "result" + str(date_set_nr).zfill(2) + ".p"
        pickle.dump((params, conf_int), open(path, "wb"))  # Pickle the Info
        
        # self.param_estimates=fit.
        # self.uncertain_estimates=fit.
    
def fac_method(method, folder, multi_processing=0):
    '''Factory Method to give the right Class which creates and analyzes the data-set'''
    if method == "testrun":
        return TestRun(folder, multi_processing=multi_processing)
    
    elif method == "multi_nbh":
        return MultiNbh(folder, multi_processing=multi_processing)

#########################################################################################
#########################################################################################       
#######################
# Some methods to test & and run this class:
######################### Some lines to test the code and make plots

def run_mult_nbh(folder):
    '''Method that can be run to simulate multiple Neighborhood Sizes'''
    MultiRun = fac_method("multi_nbh", folder)
    MultiRun.create_all_data_sets()  # Create all Datasets
    print("Creation of all Data Sets finished...")
    
    
def an_mult_nbh(folder):
    '''Analyze multiple Neighborhood Sizes'''
    MultiRun = fac_method("multi_nbh", folder)
    MultiRun.analyze_all_data_sets()
    MultiRun.save_analysis()
    print("Analysis finished and saved...")
    
def vis_mult_nbh(folder):
    '''Visualize the analysis of Multiple Neighborhood Sizes.'''
    MultiRun = fac_method("multi_nbh", folder)
    MultiRun.temp_visualize()
    
##########################################################################################
# Run all data-sets
    
if __name__ == "__main__":
    # Test = fac_method("multi_nbh", "./testfolder/")
    # Test.create_all_data_sets()  # Create all Datasets and saves them
    # print("Finished!")
    # Test.analyze_all_data_sets()  # Analyze all Datasets
    # Test.save_analysis()  # Saves the Analysis
    # Test.load_analysis()
    # print(Test.param_estimates)
    # print(Test.uncert_estimates)
    
    ####Method to Run Multiple Neighborhood Sizes:
    # run_mult_nbh("./nbh_folder/")
    
    ####Method to Analyze Multiple Neighborhood Sizes:
    # an_mult_nbh("./nbh_folder/")
    
    ####Method to Visualize Multiple Neighborhood Sizes:
    vis_mult_nbh("./nbh_folder/")
    
    



